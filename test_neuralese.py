import math
import unittest
from pathlib import Path

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
    load_benchmark_cases,
    normalize_answer,
    parse_args,
    select_benchmark_cases,
    score_answer,
    validate_cuda_environment,
)


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

    def test_benchmark_config_records_mode_and_decode_controls(self) -> None:
        args = parse_args(
            ["--benchmark", "--soft-steps", "3", "--max-new-tokens", "512"]
        )
        config = benchmark_config(args, "ordinary_cot")
        self.assertEqual(config["mode"], "ordinary_cot")
        self.assertEqual(config["gpu0_max_memory"], DEFAULT_GPU0_MAX_MEMORY)
        self.assertEqual(config["gpu1_max_memory"], DEFAULT_GPU1_MAX_MEMORY)
        self.assertEqual(config["soft_steps"], 3)
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
