from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


DEFAULT_MODEL = Path.home() / ".models/checkpoints/qwen35-9b"
DEFAULT_BENCHMARK_FILE = Path(__file__).with_name("benchmark_cases.jsonl")
DEFAULT_RESULTS_FILE = Path("benchmark_results.jsonl")
DEFAULT_GPU0_MAX_MEMORY = "15GiB"
DEFAULT_GPU1_MAX_MEMORY = "15GiB"
THINK_END = "\n</think>\n\n"
# Qwen's thinking template leaves the cursor immediately after ``<think>\n``.
# These are deliberately tokenized and fed through the cache, rather than
# included in the soft recurrence as a guessed embedding.
THINK_SCAFFOLD = "Thinking Process:\n\n"
ANSWER_MARKER_RE = re.compile(r"(?im)^\s*ANSWER\s*:\s*([^\r\n]+?)\s*$")


@dataclass
class SoftStep:
    embedding: torch.Tensor
    entropy: float
    normalized_entropy: float
    retained_mass: float
    rms: float
    top_ids: list[int]
    top_probabilities: list[float]
    effective_support: float
    temperature: float


def distribution_stats(
    logits: torch.Tensor,
    embedding_layer: torch.nn.Module,
    temperature: float,
    top_k: int,
    display_top: int = 5,
) -> SoftStep:
    """Convert next-token logits into one probability-weighted input embedding."""
    log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    probs = log_probs.exp()
    entropy_tensor = -(probs * log_probs).sum()
    vocab_size = probs.shape[-1]

    if top_k <= 0 or top_k >= vocab_size:
        selected_probs = probs
        selected_ids = torch.arange(vocab_size, device=probs.device)
        retained_mass = 1.0
    else:
        selected_probs, selected_ids = torch.topk(probs, top_k)
        retained_mass = selected_probs.sum().item()
        selected_probs = selected_probs / selected_probs.sum()

    weight = embedding_layer.weight
    selected_ids = selected_ids.to(weight.device)
    selected_probs = selected_probs.to(weight.device)

    if selected_ids.numel() == vocab_size:
        embedding = torch.matmul(
            selected_probs.to(weight.dtype).unsqueeze(0), weight
        ).float()
    else:
        vectors = embedding_layer(selected_ids).float()
        embedding = torch.sum(selected_probs.float().unsqueeze(-1) * vectors, dim=0)
        embedding = embedding.unsqueeze(0)

    shown_probs, shown_ids = torch.topk(probs, min(display_top, vocab_size))
    entropy = entropy_tensor.item()
    effective_support = math.exp(entropy)

    return SoftStep(
        embedding=embedding.unsqueeze(0).to(dtype=weight.dtype),
        entropy=entropy,
        normalized_entropy=entropy / math.log(vocab_size),
        retained_mass=retained_mass,
        rms=embedding.float().square().mean().sqrt().item(),
        top_ids=shown_ids.tolist(),
        top_probabilities=shown_probs.tolist(),
        effective_support=effective_support,
        temperature=temperature,
    )


def choose_adaptive_temperature(
    logits: torch.Tensor,
    target_support: float,
    min_temperature: float,
    max_temperature: float,
    iterations: int = 32,
) -> float:
    """Choose a bounded temperature whose effective support is near a target.

    Effective support is ``exp(entropy(softmax(logits / temperature)))``.
    Entropy is monotonic in temperature, so a deterministic binary search in
    log-temperature is sufficient and behaves well over several orders of
    magnitude.  If the requested support is outside the bounded range, the
    closest endpoint is returned.
    """
    if target_support <= 0:
        raise ValueError("target_support must be positive")
    if min_temperature <= 0 or max_temperature <= 0:
        raise ValueError("temperature bounds must be positive")
    if min_temperature > max_temperature:
        raise ValueError("min_temperature must not exceed max_temperature")
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector")

    def support(temperature: float) -> float:
        log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum().item()
        return math.exp(entropy)

    target_log_support = math.log(target_support)
    low = math.log(min_temperature)
    high = math.log(max_temperature)
    for _ in range(max(1, iterations)):
        middle = (low + high) / 2.0
        middle_temperature = math.exp(middle)
        middle_log_support = math.log(support(middle_temperature))
        if middle_log_support < target_log_support:
            low = middle
        else:
            high = middle

    candidates = [
        min_temperature,
        max_temperature,
        math.exp(low),
        math.exp(high),
    ]
    return min(candidates, key=lambda value: abs(math.log(support(value)) - target_log_support))


def choose_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0:
        return int(logits.argmax())

    scores = logits.float() / temperature
    if top_p < 1.0:
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p
        sorted_scores[remove] = -torch.inf
        scores = torch.full_like(scores, -torch.inf).scatter(0, sorted_ids, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def render_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return repr(text)


def chat_messages_input_ids(
    tokenizer,
    messages: Sequence[dict],
    use_thinking_scaffold: bool,
) -> torch.Tensor:
    """Build a chat prompt and optionally append real scaffold tokens."""
    encoded_prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    if isinstance(encoded_prompt, torch.Tensor):
        input_ids = encoded_prompt
    else:
        input_ids = encoded_prompt["input_ids"]

    if use_thinking_scaffold:
        scaffold_ids = tokenizer(
            THINK_SCAFFOLD,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        input_ids = torch.cat((input_ids, scaffold_ids), dim=-1)
    return input_ids


def chat_input_ids(tokenizer, prompt: str, use_thinking_scaffold: bool) -> torch.Tensor:
    return chat_messages_input_ids(
        tokenizer,
        [{"role": "user", "content": prompt}],
        use_thinking_scaffold,
    )


@torch.inference_mode()
def prime_thinking_context(
    model,
    tokenizer,
    prompt: str,
    use_thinking_scaffold: bool,
    messages: Sequence[dict] | None = None,
):
    if messages is None:
        input_ids = chat_input_ids(tokenizer, prompt, use_thinking_scaffold)
    else:
        input_ids = chat_messages_input_ids(tokenizer, messages, use_thinking_scaffold)
    input_ids = input_ids.to(model.device)
    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return outputs.past_key_values, outputs.logits[0, -1]


@torch.inference_mode()
def _forward_hidden_recurrence(
    model,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    past_key_values=None,
):
    """Run one hidden recurrence position through Qwen's text model only.

    Calling the causal-LM wrapper with ``output_hidden_states=True`` retains a
    hidden-state tensor for every decoder layer and every prompt position.
    That is particularly costly with Accelerate's sequential placement.  The
    Qwen 3.5 text model already returns its final (post-normalization) hidden
    state, so use it directly and project only its final position through the
    output head.  Cache objects are deliberately treated as opaque values.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("provide exactly one of input_ids or inputs_embeds")

    kwargs = {
        "use_cache": True,
        "return_dict": True,
        "past_key_values": past_key_values,
    }
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    outputs = model.model(**kwargs)
    # Clone the one position so it does not retain the base model's full
    # prompt/sequence storage (and so the next step owns its compact tensor).
    hidden = outputs.last_hidden_state[:, -1:, :].clone()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        output_embeddings = model.lm_head
    logits = output_embeddings(hidden)[:, -1, :]
    return outputs.past_key_values, logits, hidden


@torch.inference_mode()
def prime_thinking_context_with_hidden(
    model,
    tokenizer,
    prompt: str,
    use_thinking_scaffold: bool,
    messages: Sequence[dict] | None = None,
):
    """Prime the thinking context and retain its final decoder hidden state."""
    if messages is None:
        input_ids = chat_input_ids(tokenizer, prompt, use_thinking_scaffold)
    else:
        input_ids = chat_messages_input_ids(tokenizer, messages, use_thinking_scaffold)
    input_ids = input_ids.to(model.device)
    return _forward_hidden_recurrence(
        model,
        input_ids=input_ids,
    )


@torch.inference_mode()
def append_visible_anchor(model, tokenizer, cache, logits):
    anchor_ids = tokenizer(
        THINK_END,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    outputs = model(
        input_ids=anchor_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, outputs.logits[0, -1]


def _generation_eos_ids(model) -> set[int]:
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return set(eos_ids or [])


@torch.inference_mode()
def decode_visible_answer(
    model,
    tokenizer,
    cache,
    logits: torch.Tensor,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    on_text: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    generator = None
    if output_temperature > 0:
        generator = torch.Generator(device=logits.device).manual_seed(seed)

    generated: list[int] = []
    eos_ids = _generation_eos_ids(model)
    for _ in range(max_new_tokens):
        if should_stop is not None and should_stop():
            break
        token_id = choose_token(logits, output_temperature, output_top_p, generator)
        if token_id in eos_ids:
            break
        generated.append(token_id)
        if on_text is not None:
            on_text(tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
        token = torch.tensor([[token_id]], device=model.device)
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


@torch.inference_mode()
def soft_recurrent_answer(
    model,
    tokenizer,
    prompt: str,
    soft_steps: int,
    soft_temperature: float | None,
    soft_top_k: int,
    entropy_stop: float | None,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    use_thinking_scaffold: bool = True,
    target_support: float = 8.0,
    min_soft_temperature: float = 0.1,
    max_soft_temperature: float = 4.0,
    soft_top_k_warning_mass: float = 0.5,
    show_trace: bool = True,
    messages: Sequence[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_soft_step: Callable[[int, SoftStep, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    cache, logits = prime_thinking_context(
        model, tokenizer, prompt, use_thinking_scaffold, messages=messages
    )
    embedding_layer = model.get_input_embeddings()
    previous_embedding = None
    completed_steps = 0

    if show_trace:
        print("\nSoft recurrence")
    for step_index in range(soft_steps):
        if should_stop is not None and should_stop():
            break
        chosen_temperature = soft_temperature
        if chosen_temperature is None:
            chosen_temperature = choose_adaptive_temperature(
                logits,
                target_support=target_support,
                min_temperature=min_soft_temperature,
                max_temperature=max_soft_temperature,
            )
        step = distribution_stats(
            logits,
            embedding_layer,
            temperature=chosen_temperature,
            top_k=soft_top_k,
        )
        top = ", ".join(
            f"{render_token(tokenizer, token_id)}:{probability:.3f}"
            for token_id, probability in zip(step.top_ids, step.top_probabilities)
        )
        cosine = None
        if previous_embedding is not None:
            cosine = torch.nn.functional.cosine_similarity(
                previous_embedding.float().flatten(),
                step.embedding.float().flatten(),
                dim=0,
            ).item()
        cosine_text = "n/a" if cosine is None else f"{cosine:.4f}"
        if on_soft_step is not None:
            on_soft_step(step_index + 1, step, top)
        if show_trace:
            print(
                f"  {step_index + 1:02d}  temperature={step.temperature:.4f} "
                f"entropy={step.entropy:.3f} support={step.effective_support:.2f} "
                f"normalized={step.normalized_entropy:.3f} "
                f"mass={step.retained_mass:.3f} rms={step.rms:.5f} "
                f"cos(prev)={cosine_text}  {top}"
            )
            if step.retained_mass < soft_top_k_warning_mass and soft_top_k > 0:
                print(
                    f"    warning: top-k retained only {step.retained_mass:.3f} "
                    "of the probability mass; mixture may be broad"
                )

        if entropy_stop is not None and step.normalized_entropy >= entropy_stop:
            if show_trace:
                print(f"  stopped before step {step_index + 1}: entropy threshold reached")
            break

        previous_embedding = step.embedding
        outputs = model(
            inputs_embeds=step.embedding.to(model.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]
        completed_steps += 1

    cache, logits = append_visible_anchor(model, tokenizer, cache, logits)
    answer = decode_visible_answer(
        model,
        tokenizer,
        cache,
        logits,
        max_new_tokens,
        output_temperature,
        output_top_p,
        seed,
        on_text=on_text,
        should_stop=should_stop,
    )
    return answer, completed_steps


@torch.inference_mode()
def hard_argmax_answer(
    model,
    tokenizer,
    prompt: str,
    latent_steps: int,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    use_thinking_scaffold: bool = True,
    messages: Sequence[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Run latent positions by feeding the actual argmax IDs through cache."""
    cache, logits = prime_thinking_context(
        model, tokenizer, prompt, use_thinking_scaffold, messages=messages
    )
    completed_steps = 0
    for _ in range(latent_steps):
        if should_stop is not None and should_stop():
            break
        token_id = int(logits.argmax())
        token = torch.tensor([[token_id]], device=model.device)
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]
        completed_steps += 1

    cache, logits = append_visible_anchor(model, tokenizer, cache, logits)
    answer = decode_visible_answer(
        model,
        tokenizer,
        cache,
        logits,
        max_new_tokens,
        output_temperature,
        output_top_p,
        seed,
        on_text=on_text,
        should_stop=should_stop,
    )
    return answer, completed_steps


@torch.inference_mode()
def hidden_recurrent_answer(
    model,
    tokenizer,
    prompt: str,
    latent_steps: int,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    use_thinking_scaffold: bool = True,
    messages: Sequence[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_hidden_step: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Recur by feeding each latent position's final hidden state back in."""
    cache, logits, hidden = prime_thinking_context_with_hidden(
        model, tokenizer, prompt, use_thinking_scaffold, messages=messages
    )
    completed_steps = 0
    for step_index in range(latent_steps):
        if should_stop is not None and should_stop():
            break
        # A sequentially sharded model may place the final decoder output on a
        # different device from its input interface.  Moving the tensor does
        # not change its dtype or values; cache objects are passed untouched.
        inputs_embeds = hidden.to(model.device)
        cache, logits, hidden = _forward_hidden_recurrence(
            model,
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
        )
        completed_steps += 1
        rms = hidden.float().square().mean().sqrt().item()
        if on_hidden_step is not None:
            on_hidden_step(completed_steps, rms)

    cache, logits = append_visible_anchor(model, tokenizer, cache, logits)
    answer = decode_visible_answer(
        model,
        tokenizer,
        cache,
        logits,
        max_new_tokens,
        output_temperature,
        output_top_p,
        seed,
        on_text=on_text,
        should_stop=should_stop,
    )
    return answer, completed_steps


@torch.inference_mode()
def interleaved_recurrent_answer(
    model,
    tokenizer,
    prompt: str,
    hidden_steps_per_token: int,
    max_thinking_tokens: int,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    use_thinking_scaffold: bool = True,
    messages: Sequence[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_interleaved_token: Callable[[int, str, int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, int, int, bool]:
    """Alternate greedy real thinking tokens with recurrent hidden positions.

    Real tokens are kept separate from the hidden positions.  In particular,
    the closing marker is always appended as one contiguous, real-token anchor
    after this phase, even when the generated thinking tokens naturally contain
    ``THINK_END``.  This guarantees an on-manifold transition to visible answer
    decoding: natural marker pieces can otherwise have hidden positions between
    them.  If ``should_stop`` interrupts a gap, the selected token is still
    reported and the callback/counts describe the partial gap; this is the
    intentional exception to the usual ``N * (T - 1)`` hidden-position count.
    """
    if hidden_steps_per_token < 0:
        raise ValueError("hidden_steps_per_token must be non-negative")
    if max_thinking_tokens < 1:
        raise ValueError("max_thinking_tokens must be positive")

    # Prime with the ordinary CausalLM wrapper so the prompt/scaffold cache and
    # next-token logits have exactly the same shape and placement as the other
    # recurrence modes.  Each subsequent position uses the efficient text-model
    # helper, which also provides the final normalized hidden state for recurrence.
    cache, logits = prime_thinking_context(
        model, tokenizer, prompt, use_thinking_scaffold, messages=messages
    )
    thinking_ids: list[int] = []
    completed_thinking_tokens = 0
    completed_hidden_steps = 0
    thinking_end_detected = False
    eos_ids = _generation_eos_ids(model)
    final_hidden = None

    for _ in range(max_thinking_tokens):
        if should_stop is not None and should_stop():
            break

        # Thinking selection is deliberately greedy.  Sampling controls apply
        # only to the visible answer phase below.
        token_id = choose_token(logits, temperature=0.0, top_p=1.0, generator=None)
        if token_id in eos_ids:
            # EOS is a phase terminator, not a real thinking position: do not
            # feed it through the model and do not count or report it.
            break

        token = torch.tensor([[token_id]], device=model.device)
        cache, logits, final_hidden = _forward_hidden_recurrence(
            model,
            input_ids=token,
            past_key_values=cache,
        )
        completed_thinking_tokens += 1
        thinking_ids.append(token_id)

        decoded_thinking = tokenizer.decode(
            thinking_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        thinking_end_detected = THINK_END in decoded_thinking
        terminal_token = (
            thinking_end_detected
            or completed_thinking_tokens >= max_thinking_tokens
        )

        if not terminal_token:
            # Hidden positions exist only between real thinking tokens.  If a
            # stop arrives in this gap, report the completed token with the
            # state reached by the completed portion of its gap and preserve
            # both counters; no next real token is attempted.
            gap_interrupted = False
            for _ in range(hidden_steps_per_token):
                if should_stop is not None and should_stop():
                    gap_interrupted = True
                    break
                cache, logits, final_hidden = _forward_hidden_recurrence(
                    model,
                    inputs_embeds=final_hidden.to(model.device),
                    past_key_values=cache,
                )
                completed_hidden_steps += 1
            if gap_interrupted:
                terminal_token = True

        final_rms = final_hidden.float().square().mean().sqrt().item()
        if on_interleaved_token is not None:
            on_interleaved_token(
                completed_thinking_tokens,
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                completed_hidden_steps,
                final_rms,
            )
        if thinking_end_detected or terminal_token:
            break

    cache, logits = append_visible_anchor(model, tokenizer, cache, logits)
    answer = decode_visible_answer(
        model,
        tokenizer,
        cache,
        logits,
        max_new_tokens,
        output_temperature,
        output_top_p,
        seed,
        on_text=on_text,
        should_stop=should_stop,
    )
    return (
        answer,
        completed_thinking_tokens,
        completed_hidden_steps,
        thinking_end_detected,
    )


@torch.inference_mode()
def ordinary_cot_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    output_temperature: float,
    output_top_p: float,
    seed: int,
    return_visible_only: bool = True,
    messages: Sequence[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Let the model perform its ordinary visible-token thinking pass."""
    cache, logits = prime_thinking_context(
        model, tokenizer, prompt, False, messages=messages
    )
    generated = decode_visible_answer(
        model,
        tokenizer,
        cache,
        logits,
        max_new_tokens,
        output_temperature,
        output_top_p,
        seed,
        on_text=on_text,
        should_stop=should_stop,
    )
    # Keep the comparison focused on the answer while tolerating models that
    # omit the closing marker in a short generation budget.
    if return_visible_only and THINK_END in generated:
        generated = generated.split(THINK_END, 1)[1]
    return generated


def load_benchmark_cases(path: Path) -> list[dict]:
    """Load and lightly validate the hand-authored JSONL benchmark."""
    cases: list[dict] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            required = {
                "id",
                "category",
                "difficulty",
                "prompt",
                "accepted_answers",
                "rationale",
            }
            missing = required - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            if not isinstance(case["id"], str) or not case["id"]:
                raise ValueError(f"{path}:{line_number}: id must be non-empty")
            if not isinstance(case["difficulty"], str) or not case["difficulty"]:
                raise ValueError(f"{path}:{line_number}: difficulty must be non-empty")
            if case["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {case['id']}")
            if not isinstance(case["accepted_answers"], list) or not case["accepted_answers"]:
                raise ValueError(f"{path}:{line_number}: accepted_answers must be non-empty")
            seen_ids.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: benchmark contains no cases")
    return cases


def select_benchmark_cases(
    cases: Sequence[dict],
    case_ids: Sequence[str] = (),
    difficulties: Sequence[str] = (),
) -> list[dict]:
    """Apply repeatable benchmark filters and reject values absent from data."""
    known_ids = {case["id"] for case in cases}
    unknown_ids = set(case_ids) - known_ids
    if unknown_ids:
        raise ValueError(f"Unknown benchmark case(s): {', '.join(sorted(unknown_ids))}")
    known_difficulties = {case["difficulty"] for case in cases}
    unknown_difficulties = set(difficulties) - known_difficulties
    if unknown_difficulties:
        raise ValueError(
            "Unknown benchmark difficulty(s): "
            + ", ".join(sorted(unknown_difficulties))
        )
    selected = list(cases)
    if difficulties:
        difficulty_set = set(difficulties)
        selected = [case for case in selected if case["difficulty"] in difficulty_set]
    if case_ids:
        id_set = set(case_ids)
        selected = [case for case in selected if case["id"] in id_set]
    return selected


def normalize_answer(answer: str) -> str:
    """Normalize only formatting, with exact Decimal comparison for numbers."""
    value = re.sub(r"\s+", " ", str(answer).strip()).casefold()
    value = value.rstrip(".,;:!?")
    numeric = re.fullmatch(
        r"\$?\s*[+-]?(?:\d[\d,]*)(?:\.\d+)?", value
    )
    if numeric:
        # Decimal is deliberately exact: this does not accept approximate or
        # semantically equivalent answers.
        from decimal import Decimal, InvalidOperation

        try:
            decimal_value = Decimal(value.replace("$", "").replace(",", "").strip())
            return format(decimal_value.normalize(), "f")
        except InvalidOperation:
            pass
    return value


def extract_answer(raw_output: str) -> tuple[str | None, str]:
    """Extract an explicit ANSWER line; never infer an answer from prose."""
    matches = ANSWER_MARKER_RE.findall(raw_output)
    if not matches:
        return None, "missing_answer_marker"
    if len(matches) != 1:
        return None, "ambiguous_answer_marker"
    answer = matches[0].strip()
    if not answer:
        return None, "empty_answer_marker"
    return answer, "ok"


def score_answer(raw_output: str, accepted_answers: Sequence[str]) -> dict:
    extracted, status = extract_answer(raw_output)
    normalized = normalize_answer(extracted) if extracted is not None else None
    accepted = [normalize_answer(answer) for answer in accepted_answers]
    return {
        "answer": extracted,
        "normalized_answer": normalized,
        "correct": status == "ok" and normalized in accepted,
        "status": status,
    }


def benchmark_config(args: argparse.Namespace, mode: str) -> dict:
    """Return the complete, JSON-serializable configuration for one result."""
    return {
        "mode": mode,
        "model": str(args.model),
        "gpu0_max_memory": args.gpu0_max_memory,
        "gpu1_max_memory": args.gpu1_max_memory,
        "soft_steps": args.soft_steps,
        "soft_temperature": args.soft_temperature,
        "target_support": args.target_support,
        "soft_temperature_min": args.soft_temperature_min,
        "soft_temperature_max": args.soft_temperature_max,
        "soft_top_k": args.soft_top_k,
        "soft_top_k_warning_mass": args.soft_top_k_warning_mass,
        "entropy_stop": args.entropy_stop,
        "interleaved_hidden_steps": args.interleaved_hidden_steps,
        "interleaved_thinking_tokens": args.interleaved_thinking_tokens,
        "max_new_tokens": args.max_new_tokens,
        "output_temperature": args.output_temperature,
        "output_top_p": args.output_top_p,
        "seed": args.seed,
        "use_thinking_scaffold": args.use_thinking_scaffold,
    }


def run_benchmark(
    model,
    tokenizer,
    cases: Sequence[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Run all six conditions while reusing the one loaded model."""
    entropy_stop = args.entropy_stop if args.entropy_stop >= 0 else None
    common = dict(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        output_temperature=args.output_temperature,
        output_top_p=args.output_top_p,
        seed=args.seed,
    )
    results: list[dict] = []
    for case in cases:
        prompt = case["prompt"]
        mode_functions = [
            (
                "baseline",
                lambda: soft_recurrent_answer(
                    **common,
                    prompt=prompt,
                    soft_steps=0,
                    soft_temperature=args.soft_temperature,
                    soft_top_k=args.soft_top_k,
                    entropy_stop=entropy_stop,
                    use_thinking_scaffold=args.use_thinking_scaffold,
                    target_support=args.target_support,
                    min_soft_temperature=args.soft_temperature_min,
                    max_soft_temperature=args.soft_temperature_max,
                    soft_top_k_warning_mass=args.soft_top_k_warning_mass,
                    show_trace=False,
                )[0],
            ),
            (
                "hard_argmax",
                lambda: hard_argmax_answer(
                    **common,
                    prompt=prompt,
                    latent_steps=args.soft_steps,
                    use_thinking_scaffold=args.use_thinking_scaffold,
                )[0],
            ),
            (
                "soft_recurrent",
                lambda: soft_recurrent_answer(
                    **common,
                    prompt=prompt,
                    soft_steps=args.soft_steps,
                    soft_temperature=args.soft_temperature,
                    soft_top_k=args.soft_top_k,
                    entropy_stop=entropy_stop,
                    use_thinking_scaffold=args.use_thinking_scaffold,
                    target_support=args.target_support,
                    min_soft_temperature=args.soft_temperature_min,
                    max_soft_temperature=args.soft_temperature_max,
                    soft_top_k_warning_mass=args.soft_top_k_warning_mass,
                    show_trace=False,
                )[0],
            ),
            (
                "hidden_recurrent",
                lambda: hidden_recurrent_answer(
                    **common,
                    prompt=prompt,
                    latent_steps=args.soft_steps,
                    use_thinking_scaffold=args.use_thinking_scaffold,
                )[0],
            ),
            (
                "ordinary_cot",
                lambda: ordinary_cot_answer(
                    **common,
                    prompt=prompt,
                    return_visible_only=False,
                ),
            ),
            (
                "interleaved_recurrent",
                lambda: interleaved_recurrent_answer(
                    **common,
                    prompt=prompt,
                    hidden_steps_per_token=args.interleaved_hidden_steps,
                    max_thinking_tokens=args.interleaved_thinking_tokens,
                    use_thinking_scaffold=args.use_thinking_scaffold,
                )[0],
            ),
        ]
        for mode, generate in mode_functions:
            started = time.perf_counter()
            try:
                raw_output = generate()
                error_status = None
            except Exception as error:  # preserve one result per condition
                raw_output = ""
                error_status = f"execution_error: {type(error).__name__}: {error}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            scored = score_answer(raw_output, case["accepted_answers"])
            results.append({
                "case_id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "mode": mode,
                "config": benchmark_config(args, mode),
                "raw_output": raw_output,
                "extracted_answer": scored["answer"],
                "normalized_answer": scored["normalized_answer"],
                "correct": scored["correct"] if error_status is None else False,
                "status": scored["status"] if error_status is None else error_status,
                "latency_ms": round(latency_ms, 3),
            })
    return results


def write_benchmark_results(path: Path, results: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_benchmark_summary(results: Sequence[dict]) -> None:
    categories = sorted({result["category"] for result in results})
    for category in categories:
        subset = [result for result in results if result["category"] == category]
        correct = sum(bool(result["correct"]) for result in subset)
        print(f"{category}: {correct}/{len(subset)} ({correct / len(subset):.1%})")
    correct = sum(bool(result["correct"]) for result in results)
    print(f"overall: {correct}/{len(results)} ({correct / len(results):.1%})")


def build_max_memory_map(gpu0_max_memory: str, gpu1_max_memory: str) -> dict[int, str]:
    """Build an ordered GPU cap map for sequential placement.

    GPU 0 is intentionally first. Placement remains Accelerate-managed and
    never alternates layers manually.
    """
    if not isinstance(gpu0_max_memory, str) or not gpu0_max_memory.strip():
        raise ValueError("GPU max-memory values must be non-empty")
    if not isinstance(gpu1_max_memory, str) or not gpu1_max_memory.strip():
        raise ValueError("GPU max-memory values must be non-empty")
    return {0: str(gpu0_max_memory).strip(), 1: str(gpu1_max_memory).strip()}


def validate_cuda_environment(cuda_device_count: int | None = None) -> None:
    """Require the two-GPU placement policy before loading a checkpoint."""
    if cuda_device_count is None:
        cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 2:
        raise SystemExit(
            "Neuralese's default placement policy requires at least 2 CUDA GPUs "
            f"(detected {cuda_device_count}). Connect/enable a second GPU, or "
            "change the placement policy in a deployment that supports one GPU."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run probability-weighted soft tokens, recurrent hidden states, or "
            "interleaved real tokens and hidden states "
            "through Qwen before decoding an answer."
        )
    )
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--gpu0-max-memory",
        default=DEFAULT_GPU0_MAX_MEMORY,
        help="Maximum memory offered to GPU 0 (default: 15GiB).",
    )
    parser.add_argument(
        "--gpu1-max-memory",
        default=DEFAULT_GPU1_MAX_MEMORY,
        help="Maximum memory offered to GPU 1 for overflow (default: 15GiB).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the bundled six-mode benchmark instead of one prompt.",
    )
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Benchmark case ID to run; repeat to select several cases.",
    )
    parser.add_argument(
        "--difficulty",
        dest="difficulty",
        action="append",
        default=[],
        help="Benchmark difficulty to run; repeat to select several values.",
    )
    parser.add_argument("--soft-steps", type=int, default=8)
    parser.add_argument(
        "--interleaved-hidden-steps",
        type=int,
        default=2,
        help="Hidden recurrent positions between interleaved thinking tokens.",
    )
    parser.add_argument(
        "--interleaved-thinking-tokens",
        type=int,
        default=64,
        help="Maximum real thinking tokens in interleaved recurrence.",
    )
    parser.add_argument(
        "--soft-temperature",
        type=float,
        default=None,
        help="Use this fixed temperature; omit for adaptive effective-support targeting.",
    )
    parser.add_argument(
        "--soft-target-support",
        "--target-support",
        dest="target_support",
        type=float,
        default=8.0,
        help="Adaptive effective support target (default: 8).",
    )
    parser.add_argument("--soft-temperature-min", type=float, default=0.1)
    parser.add_argument("--soft-temperature-max", type=float, default=4.0)
    parser.add_argument(
        "--soft-top-k",
        type=int,
        default=64,
        help="Renormalize over this many tokens; use 0 for the full vocabulary.",
    )
    parser.add_argument(
        "--entropy-stop",
        type=float,
        default=0.75,
        help="Stop before a step at this normalized entropy; use a negative value to disable.",
    )
    parser.add_argument(
        "--soft-top-k-warning-mass",
        type=float,
        default=0.5,
        help="Warn when a retained top-k mixture contains less than this mass.",
    )
    parser.add_argument(
        "--no-thinking-scaffold",
        "--disable-thinking-scaffold",
        dest="use_thinking_scaffold",
        action="store_false",
        help="Begin soft/hard/hidden/interleaved recurrence immediately after the template's <think>\\n.",
    )
    parser.set_defaults(use_thinking_scaffold=True)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print all six baseline, recurrence, and ordinary-CoT outputs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output-temperature",
        type=float,
        default=0.0,
        help="Use 0 for greedy visible-answer decoding.",
    )
    parser.add_argument("--output-top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")
    if (
        not isinstance(args.gpu0_max_memory, str)
        or not args.gpu0_max_memory.strip()
        or not isinstance(args.gpu1_max_memory, str)
        or not args.gpu1_max_memory.strip()
    ):
        raise SystemExit("--gpu0-max-memory and --gpu1-max-memory must be non-empty.")
    if args.soft_steps < 0 or args.max_new_tokens < 1:
        raise SystemExit("Step counts must be non-negative and max-new-tokens must be positive.")
    if args.interleaved_hidden_steps < 0:
        raise SystemExit("--interleaved-hidden-steps must be non-negative.")
    if args.interleaved_thinking_tokens < 1:
        raise SystemExit("--interleaved-thinking-tokens must be positive.")
    if args.soft_temperature is not None and args.soft_temperature <= 0:
        raise SystemExit("--soft-temperature must be positive.")
    if args.target_support <= 0:
        raise SystemExit("--soft-target-support must be positive.")
    if args.soft_temperature_min <= 0 or args.soft_temperature_max <= 0:
        raise SystemExit("Soft temperature bounds must be positive.")
    if args.soft_temperature_min > args.soft_temperature_max:
        raise SystemExit("--soft-temperature-min must not exceed --soft-temperature-max.")
    if not 0 <= args.soft_top_k_warning_mass <= 1:
        raise SystemExit("--soft-top-k-warning-mass must be in [0, 1].")
    if args.soft_top_k < 0:
        raise SystemExit("--soft-top-k must be non-negative.")
    if not 0 < args.output_top_p <= 1:
        raise SystemExit("--output-top-p must be in (0, 1].")
    if not args.benchmark and not args.prompt:
        raise SystemExit("A prompt is required unless --benchmark is used.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)

    benchmark_cases = None
    if args.benchmark:
        try:
            benchmark_cases = select_benchmark_cases(
                load_benchmark_cases(args.benchmark_file),
                case_ids=args.case_ids,
                difficulties=args.difficulty,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    validate_cuda_environment()
    max_memory = build_max_memory_map(
        args.gpu0_max_memory,
        args.gpu1_max_memory,
    )
    print(f"Loading {args.model} across available GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="sequential",
        max_memory=max_memory,
        local_files_only=True,
    )
    model.eval()

    if args.benchmark:
        results = run_benchmark(model, tokenizer, benchmark_cases, args)
        write_benchmark_results(args.results_file, results)
        print_benchmark_summary(results)
        print(f"wrote {len(results)} results to {args.results_file}")
        return

    entropy_stop = args.entropy_stop if args.entropy_stop >= 0 else None
    common = dict(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        output_temperature=args.output_temperature,
        output_top_p=args.output_top_p,
        seed=args.seed,
    )
    if args.compare:
        baseline, _ = soft_recurrent_answer(
            **common,
            soft_steps=0,
            soft_temperature=args.soft_temperature,
            soft_top_k=args.soft_top_k,
            entropy_stop=entropy_stop,
            use_thinking_scaffold=args.use_thinking_scaffold,
            target_support=args.target_support,
            min_soft_temperature=args.soft_temperature_min,
            max_soft_temperature=args.soft_temperature_max,
            soft_top_k_warning_mass=args.soft_top_k_warning_mass,
            show_trace=False,
        )
        hard, hard_steps = hard_argmax_answer(
            **common,
            latent_steps=args.soft_steps,
            use_thinking_scaffold=args.use_thinking_scaffold,
        )
        soft, soft_completed = soft_recurrent_answer(
            **common,
            soft_steps=args.soft_steps,
            soft_temperature=args.soft_temperature,
            soft_top_k=args.soft_top_k,
            entropy_stop=entropy_stop,
            use_thinking_scaffold=args.use_thinking_scaffold,
            target_support=args.target_support,
            min_soft_temperature=args.soft_temperature_min,
            max_soft_temperature=args.soft_temperature_max,
            soft_top_k_warning_mass=args.soft_top_k_warning_mass,
            show_trace=True,
        )
        hidden, hidden_completed = hidden_recurrent_answer(
            **common,
            latent_steps=args.soft_steps,
            use_thinking_scaffold=args.use_thinking_scaffold,
        )
        cot = ordinary_cot_answer(**common)
        interleaved, interleaved_thinking, interleaved_hidden, interleaved_end = (
            interleaved_recurrent_answer(
                **common,
                hidden_steps_per_token=args.interleaved_hidden_steps,
                max_thinking_tokens=args.interleaved_thinking_tokens,
                use_thinking_scaffold=args.use_thinking_scaffold,
            )
        )
        print(f"\n=== No latent steps (baseline) ===\n{baseline}")
        print(f"\n=== Hard argmax steps ({hard_steps}) ===\n{hard}")
        print(f"\n=== Soft recurrent steps ({soft_completed}) ===\n{soft}")
        print(f"\n=== Hidden recurrent steps ({hidden_completed}) ===\n{hidden}")
        print(f"\n=== Ordinary visible CoT ===\n{cot}")
        print(
            f"\n=== Interleaved recurrent tokens ({interleaved_thinking}), "
            f"hidden steps ({interleaved_hidden}), "
            f"thinking end ({interleaved_end}) ===\n{interleaved}"
        )
    else:
        answer, completed_steps = soft_recurrent_answer(
            **common,
            soft_steps=args.soft_steps,
            soft_temperature=args.soft_temperature,
            soft_top_k=args.soft_top_k,
            entropy_stop=entropy_stop,
            use_thinking_scaffold=args.use_thinking_scaffold,
            target_support=args.target_support,
            min_soft_temperature=args.soft_temperature_min,
            max_soft_temperature=args.soft_temperature_max,
            soft_top_k_warning_mass=args.soft_top_k_warning_mass,
        )
        print(f"\nAnswer after {completed_steps} soft steps\n{answer}")


if __name__ == "__main__":
    main()
