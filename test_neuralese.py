import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from neuralese import (
    DEFAULT_GPU0_MAX_MEMORY,
    DEFAULT_GPU1_MAX_MEMORY,
    THINK_SCAFFOLD,
    benchmark_config,
    build_max_memory_map,
    choose_adaptive_temperature,
    choose_token,
    distribution_stats,
    extract_answer,
    hidden_recurrent_answer,
    interleaved_recurrent_answer,
    load_benchmark_cases,
    normalize_answer,
    parse_args,
    run_benchmark,
    select_benchmark_cases,
    score_answer,
    validate_cuda_environment,
    validate_args,
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking, return_tensors
    ):
        return torch.tensor([[1, 2]])

    def __call__(self, text, add_special_tokens, return_tensors):
        if text == "Thinking Process:\n\n":
            ids = [3, 4]
        elif text == "\n</think>\n\n":
            ids = [8, 9]
        else:
            raise AssertionError(f"unexpected tokenizer input: {text!r}")
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, token_ids, **kwargs):
        return "ANSWER: 42" if token_ids else ""


class FakeOutputEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(100, 3))

    def forward(self, hidden):
        logits = torch.zeros(*hidden.shape[:-1], 100)
        # Keep this a genuine rank-[batch, sequence, vocab] output.  The
        # actual hidden recurrence only needs the projection's final position.
        logits[..., 1] = 10.0
        return logits


class FakeTextModel:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, **kwargs):
        self.owner.base_calls.append(kwargs)
        self.owner._cache_number += 1
        if "input_ids" in kwargs:
            ids = kwargs["input_ids"]
            # Every prompt position is distinct, making accidental selection
            # of the first position observable.  The scaffold's final ID is 4.
            values = ids.float() * 10 + torch.arange(ids.shape[1]).float()
            last_value = values[:, -1]
            full_hidden = values.unsqueeze(-1).expand(-1, -1, 3).clone()
        else:
            previous = kwargs["inputs_embeds"][:, -1, :].mean(dim=-1)
            last_value = previous + 1
            full_hidden = last_value[:, None, None].expand(-1, 1, 3).clone()
        self.owner.last_full_hidden = full_hidden
        self.owner.full_hidden_history.append(full_hidden)
        return SimpleNamespace(
            past_key_values=f"base-cache-{self.owner._cache_number}",
            last_hidden_state=full_hidden,
        )


class FakeHiddenModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.calls = []
        self.base_calls = []
        self._cache_number = 0
        self.last_full_hidden = None
        self.full_hidden_history = []
        self.model = FakeTextModel(self)
        self.output_embeddings = FakeOutputEmbeddings()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        logits = torch.zeros(1, input_ids.shape[1], 100)
        if int(input_ids[0, -1]) == 9:  # visible anchor
            logits[:, :, 1] = 10.0
        else:  # token-sensitive visible decoding: terminate after one token
            logits[:, :, 99] = 10.0
        return SimpleNamespace(
            past_key_values="visible-cache",
            logits=logits,
        )

    def get_output_embeddings(self):
        return self.output_embeddings


class HiddenRecurrentTests(unittest.TestCase):
    def test_recurrence_uses_final_hidden_states_and_forwards_cache(self):
        model = FakeHiddenModel()
        tokenizer = FakeTokenizer()
        callbacks = []

        answer, completed = hidden_recurrent_answer(
            model,
            tokenizer,
            "question",
            latent_steps=2,
            max_new_tokens=1,
            output_temperature=0.0,
            output_top_p=1.0,
            seed=0,
            on_hidden_step=lambda index, rms: callbacks.append((index, rms)),
        )

        self.assertEqual(answer, "ANSWER: 42")
        self.assertEqual(completed, 2)
        recurrent_calls = [call for call in model.base_calls if "inputs_embeds" in call]
        self.assertEqual(len(recurrent_calls), 2)
        torch.testing.assert_close(
            recurrent_calls[0]["inputs_embeds"], torch.full((1, 1, 3), 43.0)
        )
        torch.testing.assert_close(
            recurrent_calls[1]["inputs_embeds"], torch.full((1, 1, 3), 44.0)
        )
        self.assertEqual(recurrent_calls[0]["past_key_values"], "base-cache-1")
        self.assertEqual(recurrent_calls[1]["past_key_values"], "base-cache-2")
        self.assertEqual(callbacks[0][0], 1)
        self.assertAlmostEqual(callbacks[0][1], 44.0)
        self.assertEqual(callbacks[1][0], 2)
        self.assertAlmostEqual(callbacks[1][1], 45.0)
        self.assertTrue(all(call["use_cache"] and call["return_dict"] for call in model.base_calls))
        self.assertTrue(all("output_hidden_states" not in call for call in model.base_calls))
        prompt_hidden = model.full_hidden_history[0]
        self.assertLess(recurrent_calls[0]["inputs_embeds"].numel(), prompt_hidden.numel())
        self.assertNotEqual(
            prompt_hidden.untyped_storage().data_ptr(),
            recurrent_calls[0]["inputs_embeds"].untyped_storage().data_ptr(),
        )

        # The anchor is a real visible token sequence after the latent cache,
        # and visible decoding uses its logits rather than latent logits.
        anchor = next(call for call in model.calls if "input_ids" in call)
        self.assertEqual(anchor["input_ids"].tolist(), [[8, 9]])
        self.assertEqual(anchor["past_key_values"], "base-cache-3")
        self.assertEqual(model.calls[-1]["input_ids"].tolist(), [[1]])

    def test_stop_and_zero_steps_still_anchor_and_decode(self):
        model = FakeHiddenModel()
        tokenizer = FakeTokenizer()
        stop = lambda: True

        answer, completed = hidden_recurrent_answer(
            model,
            tokenizer,
            "question",
            latent_steps=3,
            max_new_tokens=1,
            output_temperature=0.0,
            output_top_p=1.0,
            seed=0,
            should_stop=stop,
        )
        self.assertEqual((answer, completed), ("", 0))
        self.assertEqual(sum("inputs_embeds" in call for call in model.base_calls), 0)
        self.assertTrue(any("past_key_values" in call for call in model.calls))

        # A stop observed after one completed latent position prevents the
        # next position, while still preserving the completed-step count.
        model = FakeHiddenModel()
        checks = 0

        def stop_after_one():
            nonlocal checks
            checks += 1
            return checks >= 2

        answer, completed = hidden_recurrent_answer(
            model, tokenizer, "question", latent_steps=3, max_new_tokens=1,
            output_temperature=0.0, output_top_p=1.0, seed=0,
            should_stop=stop_after_one,
        )
        self.assertEqual((answer, completed), ("", 1))
        self.assertEqual(len(model.base_calls), 2)

        model = FakeHiddenModel()
        answer, completed = hidden_recurrent_answer(
            model,
            tokenizer,
            "question",
            latent_steps=0,
            max_new_tokens=1,
            output_temperature=0.0,
            output_top_p=1.0,
            seed=0,
        )
        self.assertEqual((answer, completed), ("ANSWER: 42", 0))
        self.assertEqual(sum("inputs_embeds" in call for call in model.base_calls), 0)


class InterleavedTokenizer(FakeTokenizer):
    def __init__(self, natural_end=False):
        self.natural_end = natural_end

    def decode(self, token_ids, **kwargs):
        ids = list(token_ids)
        if self.natural_end and ids == [10, 11]:
            return "x" + "\n</think>\n\n"
        if ids == [1]:
            return "ANSWER: 42"
        names = {10: "a", 11: "b", 12: "c"}
        return "".join(names.get(token_id, "") for token_id in ids)


class InterleavedOutputEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(100, 3))

    def forward(self, hidden):
        logits = torch.full((*hidden.shape[:-1], 100), -100.0)
        states = hidden[..., 0].round().long()
        # The logits immediately after real10/real11 deliberately point at a
        # different token than the logits after their hidden gaps.  A stale
        # real-token logit would therefore select 13 instead of the expected
        # next token.
        for state, token_id in ((10, 13), (11, 13), (12, 11), (13, 12)):
            logits[..., token_id] = torch.where(
                states == state, torch.tensor(10.0), logits[..., token_id]
            )
        logits[..., 99] = torch.where(
            states >= 14, torch.tensor(10.0), logits[..., 99]
        )
        return logits


class InterleavedModel(FakeHiddenModel):
    def __init__(self):
        super().__init__()
        self.output_embeddings = InterleavedOutputEmbeddings()
        self.trace = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "input_ids" in kwargs and "past_key_values" not in kwargs:
            self.trace.append(("prime", kwargs["input_ids"].tolist(), None))
            return SimpleNamespace(past_key_values="prompt-cache", logits=self._prime_logits())
        if "input_ids" in kwargs:
            is_anchor = int(kwargs["input_ids"][0, -1]) == 9
            self.trace.append((
                "anchor" if is_anchor else "visible",
                int(kwargs["input_ids"][0, 0]),
                kwargs.get("past_key_values"),
            ))
            logits = torch.zeros(1, kwargs["input_ids"].shape[1], 100)
            if is_anchor:
                logits[:, :, 1] = 10.0
            else:
                logits[:, :, 99] = 10.0
            return SimpleNamespace(past_key_values="visible-cache", logits=logits)
        raise AssertionError("unexpected wrapper call")

    @staticmethod
    def _prime_logits():
        logits = torch.full((1, 2, 100), -100.0)
        logits[:, -1, 10] = 10.0
        return logits


class InterleavedTextModel(FakeTextModel):
    def __call__(self, **kwargs):
        self.owner.base_calls.append(kwargs)
        self.owner._cache_number += 1
        if "input_ids" in kwargs:
            token_id = int(kwargs["input_ids"][0, -1])
            value = float(token_id)
            kind = "real"
        else:
            # Each latent position advances the recurrent state, so logits
            # from the preceding real token are observably stale.
            value = float(kwargs["inputs_embeds"][0, -1, 0]) + 1
            kind = "hidden"
        self.owner.trace.append((kind, int(value), kwargs.get("past_key_values")))
        hidden = torch.full((1, 1, 3), value)
        return SimpleNamespace(
            past_key_values=f"cache-{self.owner._cache_number}",
            last_hidden_state=hidden,
        )


class InterleavedTests(unittest.TestCase):
    def test_token_first_schedule_uses_final_latent_logits_and_contiguous_anchor(self):
        model = InterleavedModel()
        model.model = InterleavedTextModel(model)
        callbacks = []
        answer, thinking, hidden, ended = interleaved_recurrent_answer(
            model, InterleavedTokenizer(), "question", 2, 3, 1, 0.0, 1.0, 0,
            on_interleaved_token=lambda *event: callbacks.append(event),
        )
        self.assertEqual((answer, thinking, hidden, ended), ("ANSWER: 42", 3, 4, False))
        self.assertEqual(
            [(kind, value) for kind, value, _ in model.trace[1:] if kind != "visible"],
            [("real", 10), ("hidden", 11), ("hidden", 12),
             ("real", 11), ("hidden", 12), ("hidden", 13), ("real", 12),
             ("anchor", 8)],
        )
        self.assertEqual([event[:3] for event in callbacks], [(1, "a", 2), (2, "b", 4), (3, "c", 4)])
        self.assertEqual([event[3] for event in callbacks], [12.0, 13.0, 12.0])
        self.assertEqual(model.trace[1][2], "prompt-cache")
        self.assertEqual(model.trace[4][2], "cache-3")
        self.assertEqual(model.trace[7][2], "cache-6")
        self.assertEqual(model.trace[8][2], "cache-7")

    def test_natural_end_stops_before_gap_but_still_appends_anchor(self):
        model = InterleavedModel()
        model.model = InterleavedTextModel(model)
        tokenizer = InterleavedTokenizer(natural_end=True)
        answer, thinking, hidden, ended = interleaved_recurrent_answer(
            model, tokenizer, "question", 2, 4, 1, 0.0, 1.0, 0,
        )
        self.assertEqual((answer, thinking, hidden, ended), ("ANSWER: 42", 2, 2, True))
        self.assertEqual([kind for kind, _, _ in model.trace[1:] if kind != "visible"],
                         ["real", "hidden", "hidden", "real", "anchor"])
        self.assertEqual(next(value for kind, value, _ in reversed(model.trace) if kind == "anchor"), 8)

    def test_stop_in_gap_and_eos_do_not_consume_extra_positions(self):
        model = InterleavedModel()
        model.model = InterleavedTextModel(model)
        checks = 0

        def stop_in_second_hidden():
            nonlocal checks
            checks += 1
            return checks == 3  # before first token, before hidden 1, before hidden 2

        answer, thinking, hidden, ended = interleaved_recurrent_answer(
            model, InterleavedTokenizer(), "question", 2, 4, 1, 0.0, 1.0, 0,
            should_stop=stop_in_second_hidden,
        )
        self.assertEqual((answer, thinking, hidden, ended), ("ANSWER: 42", 1, 1, False))
        self.assertEqual([kind for kind, _, _ in model.trace[1:] if kind != "visible"], ["real", "hidden", "anchor"])

        model = InterleavedModel()
        model.model = InterleavedTextModel(model)
        model.output_embeddings = InterleavedOutputEmbeddings()
        # Make the first real token's logits select EOS on the next iteration.
        model.output_embeddings.forward = lambda hidden: torch.cat(
            (torch.full((*hidden.shape[:-1], 99), -100.0),
             torch.full((*hidden.shape[:-1], 1), 10.0)), dim=-1
        )
        answer, thinking, hidden, ended = interleaved_recurrent_answer(
            model, InterleavedTokenizer(), "question", 2, 4, 1, 0.0, 1.0, 0,
        )
        self.assertEqual((answer, thinking, hidden, ended), ("ANSWER: 42", 1, 2, False))
        self.assertEqual([kind for kind, _, _ in model.trace[1:] if kind != "visible"],
                         ["real", "hidden", "hidden", "anchor"])


class BenchmarkTests(unittest.TestCase):
    def test_run_benchmark_keeps_six_modes_in_order_after_execution_error(self):
        args = parse_args(["--benchmark", "--soft-steps", "2"])
        case = {
            "id": "demo",
            "category": "logic_constraints",
            "difficulty": "calibration",
            "prompt": "question",
            "accepted_answers": ["42"],
            "rationale": "test",
        }

        def soft(**kwargs):
            return "ANSWER: 42", 0

        def hard(**kwargs):
            return "ANSWER: 42", 2

        def hidden(**kwargs):
            raise RuntimeError("hidden failed")

        def interleaved(**kwargs):
            return "ANSWER: 42", 2, 4, False

        with patch("neuralese.soft_recurrent_answer", side_effect=soft), \
             patch("neuralese.hard_argmax_answer", side_effect=hard), \
             patch("neuralese.hidden_recurrent_answer", side_effect=hidden), \
             patch("neuralese.interleaved_recurrent_answer", side_effect=interleaved), \
             patch("neuralese.ordinary_cot_answer", return_value="ANSWER: 42"):
            results = run_benchmark(object(), object(), [case], args)

        self.assertEqual(
            [result["mode"] for result in results],
            [
                "baseline", "hard_argmax", "soft_recurrent", "hidden_recurrent",
                "ordinary_cot", "interleaved_recurrent",
            ],
        )
        self.assertEqual(results[3]["status"], "execution_error: RuntimeError: hidden failed")
        self.assertEqual(results[4]["raw_output"], "ANSWER: 42")
        self.assertEqual(results[5]["raw_output"], "ANSWER: 42")


class DistributionStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = torch.nn.Embedding(3, 2)
        self.embedding.weight.data.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        )
        self.logits = torch.log(torch.tensor([0.2, 0.3, 0.5]))

    def test_full_distribution_is_weighted_average(self) -> None:
        step = distribution_stats(self.logits, self.embedding, 1.0, top_k=0)

        expected = torch.tensor([[[1.2, 1.3]]])
        torch.testing.assert_close(step.embedding, expected)
        self.assertAlmostEqual(step.retained_mass, 1.0)
        self.assertAlmostEqual(
            step.entropy,
            -(0.2 * math.log(0.2) + 0.3 * math.log(0.3) + 0.5 * math.log(0.5)),
            places=6,
        )
        self.assertAlmostEqual(step.effective_support, math.exp(step.entropy), places=6)
        self.assertEqual(step.temperature, 1.0)

    def test_top_k_is_renormalized(self) -> None:
        step = distribution_stats(self.logits, self.embedding, 1.0, top_k=2)

        expected = torch.tensor([[[1.25, 1.625]]])
        torch.testing.assert_close(step.embedding, expected)
        self.assertAlmostEqual(step.retained_mass, 0.8, places=6)

    def test_fixed_temperature_reports_fixed_temperature_and_support(self) -> None:
        step = distribution_stats(self.logits, self.embedding, 2.0, top_k=0)
        self.assertEqual(step.temperature, 2.0)
        self.assertGreater(step.effective_support, 1.0)

    def test_adaptive_temperature_targets_effective_support(self) -> None:
        logits = torch.tensor([8.0, 4.0, 0.0, -4.0, -8.0])
        temperature = choose_adaptive_temperature(logits, 2.0, 0.1, 20.0)
        embedding = torch.nn.Embedding(5, 2)
        support = distribution_stats(logits, embedding, temperature, top_k=0).effective_support
        self.assertGreaterEqual(temperature, 0.1)
        self.assertLessEqual(temperature, 20.0)
        self.assertAlmostEqual(support, 2.0, delta=0.02)

    def test_adaptive_temperature_clamps_unreachable_target(self) -> None:
        temperature = choose_adaptive_temperature(self.logits, 100.0, 0.1, 2.0)
        self.assertEqual(temperature, 2.0)

    def test_greedy_selection(self) -> None:
        selected = choose_token(self.logits, temperature=0, top_p=1, generator=None)
        self.assertEqual(selected, 2)


class CliTests(unittest.TestCase):
    def test_adaptive_temperature_and_scaffold_defaults_need_no_model(self) -> None:
        args = parse_args(["question"])
        self.assertIsNone(args.soft_temperature)
        self.assertEqual(args.target_support, 8.0)
        self.assertTrue(args.use_thinking_scaffold)
        self.assertFalse(parse_args(["question", "--no-thinking-scaffold"]).use_thinking_scaffold)
        self.assertEqual(THINK_SCAFFOLD, "Thinking Process:\n\n")

    def test_gpu_memory_defaults_and_overrides(self) -> None:
        defaults = parse_args(["question"])
        self.assertEqual(defaults.gpu0_max_memory, DEFAULT_GPU0_MAX_MEMORY)
        self.assertEqual(defaults.gpu1_max_memory, DEFAULT_GPU1_MAX_MEMORY)
        overridden = parse_args(
            ["question", "--gpu0-max-memory", "14GiB", "--gpu1-max-memory", "5GiB"]
        )
        self.assertEqual(
            build_max_memory_map(overridden.gpu0_max_memory, overridden.gpu1_max_memory),
            {0: "14GiB", 1: "5GiB"},
        )
        self.assertEqual(list(build_max_memory_map("15GiB", "15GiB")), [0, 1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_max_memory_map("", "6GiB")

    def test_cuda_policy_validation_is_pure(self) -> None:
        validate_cuda_environment(2)
        with self.assertRaisesRegex(SystemExit, "at least 2 CUDA GPUs"):
            validate_cuda_environment(1)

    def test_compare_mode_parses_latent_and_decode_budgets(self) -> None:
        args = parse_args(
            ["question", "--compare", "--soft-steps", "3", "--max-new-tokens", "11"]
        )
        self.assertTrue(args.compare)
        self.assertEqual(args.soft_steps, 3)
        self.assertEqual(args.max_new_tokens, 11)

    def test_interleaved_defaults_and_validation(self) -> None:
        args = parse_args(["question"])
        self.assertEqual(args.interleaved_hidden_steps, 2)
        self.assertEqual(args.interleaved_thinking_tokens, 64)
        args.model = Path(".")
        validate_args(args)
        invalid_gap = parse_args(["question", "--interleaved-hidden-steps", "-1"])
        invalid_gap.model = Path(".")
        with self.assertRaisesRegex(SystemExit, "interleaved-hidden-steps"):
            validate_args(invalid_gap)
        invalid_tokens = parse_args(["question", "--interleaved-thinking-tokens", "0"])
        invalid_tokens.model = Path(".")
        with self.assertRaisesRegex(SystemExit, "interleaved-thinking-tokens"):
            validate_args(invalid_tokens)

    def test_benchmark_file_has_four_categories_and_unique_cases(self) -> None:
        cases = load_benchmark_cases(Path(__file__).with_name("benchmark_cases.jsonl"))
        self.assertEqual(len(cases), 28)
        self.assertEqual(
            {case["difficulty"] for case in cases}, {"calibration", "challenge"}
        )
        self.assertEqual(
            sum(case["difficulty"] == "calibration" for case in cases), 16
        )
        self.assertEqual(
            sum(case["difficulty"] == "challenge" for case in cases), 12
        )
        categories = {case["category"] for case in cases}
        self.assertEqual(
            categories,
            {
                "arithmetic_units",
                "state_tracking",
                "logic_constraints",
                "symbolic_algorithms",
            },
        )
        for category in categories:
            self.assertEqual(sum(case["category"] == category for case in cases), 7)
        self.assertEqual(len(cases), len({case["id"] for case in cases}))

    def test_numerical_state_and_symbolic_answers_independently(self) -> None:
        cases = {
            case["id"]: case
            for case in load_benchmark_cases(Path(__file__).with_name("benchmark_cases.jsonl"))
        }
        robot_x, robot_y = 2, -1
        for dx, dy in [(0, 4), (-3, 0), (0, -2), (5, 0)]:
            robot_x, robot_y = robot_x + dx, robot_y + dy
        challenge_x, challenge_y = 3, -2
        for dx, dy in [(0, 5), (4, 0), (0, -3), (-2, 0), (0, 1)]:
            challenge_x, challenge_y = challenge_x + dx, challenge_y + dy
        challenge_x, challenge_y = -challenge_x + 2, challenge_y - 3
        symbolic_string = ""
        seen = set()
        for char in "ABCA":
            symbolic_string += char if char not in seen else "X"
            seen.add(char)
        symbolic_values = [value * value - 1 for value in [1, 3, 4, 6]]
        symbolic_values = [value for value in symbolic_values if value % 3]
        symbolic_values.reverse()
        running = 0
        cumulative = []
        for value in symbolic_values:
            running += value
            cumulative.append(running)
        recurrence = 3
        for _ in range(4):
            recurrence = 2 * recurrence + 1
        digits = [(3 * value + 2) % 10 for value in [2, 7, 1, 8, 2]]
        digits = digits[-2:] + digits[:-2]
        digits = list(dict.fromkeys(digits))
        symbolic_digits = "".join(str(value) for value in reversed(digits))
        expected = {
            "arith_01": (8 * 12 - 17) // 5,
            "arith_02": 2400 - 350 + 750 - 800,
            "arith_03": (3.2 + 1.8) * 2.5 + 0.6,
            "arith_04": round(80 * 0.85 * 1.10, 1),
            "arith_05": 35 + 18 * 7 - 2.5 * 4 - 3.2,
            "arith_06": int(2400 + 18 * 27.5 - 350 + 3 * 125),
            "arith_07": int((18 / 18 + 12 / 24 + 12 / 15) * 60),
            "state_01": (12 + 7) * 2 - 5,
            "state_02": (14 - 3) + (9 + 3 - 5),
            "state_03": f"({robot_x},{robot_y})",
            "state_04": 75 - 12 + 7 - 3,
            "state_05": (120 - 28 + 17 - 6) + (75 + 28 - 33) - (40 - 17 + 33),
            "state_06": f"({challenge_x},{challenge_y})",
            "state_07": (48 - 9 - 15 + 6) + (31 + 9 - 7) - (20 + 7 + 12 - 6),
            "symbolic_01": sum(2 * value + 1 for value in [3, 1, 4, 1, 5]),
            "symbolic_02": symbolic_string,
            "symbolic_03": 2 + 2**2 + 3**2 + 4**2,
            "symbolic_04": sum(value + 10 for value in [1, 3, 5]),
            "symbolic_05": cumulative[-1],
            "symbolic_06": (recurrence - 7) // 4 + 6,
            "symbolic_07": symbolic_digits,
        }
        self.assertEqual(len(expected), 21)
        for case_id, answer in expected.items():
            self.assertIn(str(answer), [str(value) for value in cases[case_id]["accepted_answers"]])

    def test_answer_scoring_is_marker_only_and_exact(self) -> None:
        self.assertEqual(extract_answer("work\nANSWER: $1,200.00\n"), ("$1,200.00", "ok"))
        self.assertEqual(normalize_answer("$1,200.00!"), "1200")
        self.assertTrue(score_answer("ANSWER: $1,200.00", ["1200"])["correct"])
        self.assertFalse(score_answer("The answer is 1200.", ["1200"])["correct"])
        self.assertEqual(score_answer("ANSWER: 12", ["13"])["status"], "ok")
        self.assertEqual(score_answer("no marker", ["12"])["status"], "missing_answer_marker")

    def test_logic_06_accepts_spaced_pairs_and_requires_answer_marker(self) -> None:
        cases = load_benchmark_cases(Path(__file__).with_name("benchmark_cases.jsonl"))
        logic_06 = next(case for case in cases if case["id"] == "logic_06")
        accepted_answers = logic_06["accepted_answers"]

        self.assertTrue(score_answer("ANSWER: P, Q", accepted_answers)["correct"])
        self.assertTrue(score_answer("ANSWER: Q, P", accepted_answers)["correct"])
        self.assertFalse(score_answer("ANSWER: P/R", accepted_answers)["correct"])
        self.assertEqual(
            score_answer("The answer is P, Q.", accepted_answers)["status"],
            "missing_answer_marker",
        )

    def test_benchmark_config_records_mode_and_decode_controls(self) -> None:
        args = parse_args(
            ["--benchmark", "--soft-steps", "3", "--max-new-tokens", "512"]
        )
        config = benchmark_config(args, "ordinary_cot")
        self.assertEqual(config["mode"], "ordinary_cot")
        self.assertEqual(config["gpu0_max_memory"], DEFAULT_GPU0_MAX_MEMORY)
        self.assertEqual(config["gpu1_max_memory"], DEFAULT_GPU1_MAX_MEMORY)
        self.assertEqual(config["soft_steps"], 3)
        self.assertEqual(config["interleaved_hidden_steps"], 2)
        self.assertEqual(config["interleaved_thinking_tokens"], 64)
        self.assertEqual(config["max_new_tokens"], 512)

    def test_difficulty_filter_is_repeatable_and_rejects_unknown_values(self) -> None:
        args = parse_args(
            ["--benchmark", "--difficulty", "challenge", "--difficulty", "calibration"]
        )
        self.assertEqual(args.difficulty, ["challenge", "calibration"])
        cases = load_benchmark_cases(Path(__file__).with_name("benchmark_cases.jsonl"))
        challenge = select_benchmark_cases(cases, difficulties=["challenge"])
        self.assertEqual(len(challenge), 12)
        with self.assertRaisesRegex(ValueError, "Unknown benchmark difficulty"):
            select_benchmark_cases(cases, difficulties=["extreme"])


if __name__ == "__main__":
    unittest.main()
