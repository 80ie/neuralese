from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import torch
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.utils import logging as transformers_logging

from neuralese import (
    DEFAULT_BENCHMARK_FILE,
    DEFAULT_GPU0_MAX_MEMORY,
    DEFAULT_GPU1_MAX_MEMORY,
    DEFAULT_MODEL,
    build_max_memory_map,
    hard_argmax_answer,
    load_benchmark_cases,
    ordinary_cot_answer,
    score_answer,
    select_benchmark_cases,
    soft_recurrent_answer,
    validate_cuda_environment,
)


MODES = ("baseline", "hard_argmax", "soft_recurrent", "ordinary_cot")
MODE_LABELS = {
    "baseline": "Baseline",
    "hard_argmax": "Hard argmax",
    "soft_recurrent": "Soft recurrent",
    "ordinary_cot": "Ordinary CoT",
}


def parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def parse_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def summarize_results(results: Sequence[dict]) -> list[tuple[str, int, int, float, float, float]]:
    rows = []
    for mode in MODES:
        subset = [result for result in results if result["mode"] == mode]
        correct = sum(bool(result["correct"]) for result in subset)
        accuracy = correct / len(subset) if subset else 0.0
        latency = (
            sum(float(result["latency_ms"]) for result in subset) / len(subset)
            if subset
            else 0.0
        )
        throughput = (
            sum(float(result.get("tokens_per_second", 0.0)) for result in subset)
            / len(subset)
            if subset
            else 0.0
        )
        rows.append((mode, correct, len(subset), accuracy, latency, throughput))
    return rows


class NeuraleseTUI(App):
    TITLE = "Neuralese Harness"
    SUB_TITLE = "Soft-token recurrence lab"
    CSS = """
    Screen { background: #111318; color: #e7e9ee; }
    Header { background: #542f1f; }
    #status-bar { height: 1; padding: 0 1; background: #28231f; color: #e8b68f; }
    TabbedContent { height: 1fr; }
    .toolbar { height: 3; padding: 0 1; align-vertical: middle; }
    .toolbar Label { width: auto; margin: 0 1 0 0; }
    .small-input { width: 9; margin-right: 1; }
    .medium-select { width: 18; margin-right: 1; }
    Button { margin-right: 1; min-width: 10; }
    #chat-layout { height: 1fr; grid-size: 2 1; grid-columns: 2fr 1fr; grid-gutter: 1; }
    #chat-main, #chat-side { height: 1fr; }
    #chat-transcript { height: 1fr; border: round #6f4a35; padding: 0 1; }
    #chat-live { height: 12; border: round #b0673f; }
    #chat-trace { height: 1fr; border: round #6f4a35; }
    #chat-entry { height: 3; }
    #chat-input { width: 1fr; }
    #benchmark-layout { height: 1fr; grid-size: 2 2; grid-columns: 3fr 2fr; grid-rows: 2fr 3fr; grid-gutter: 0; padding: 0 1 1 1; }
    #case-table { height: 1fr; border: round #6f4a35; }
    #summary-table { height: 1fr; border: round #6f4a35; }
    #output-grid { column-span: 2; height: 1fr; grid-size: 2 2; grid-gutter: 0; }
    .output-panel { height: 1fr; border: round #6f4a35; }
    .output-title { height: 1; padding: 0 1; background: #28231f; color: #e8b68f; }
    .output-text { height: 1fr; }
    #benchmark-progress { width: 24; margin-right: 1; }
    #benchmark-help { height: 1; padding: 0 1; color: #aaa39b; }
    """
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+b", "start_benchmark", "Run benchmark"),
        ("escape", "request_stop", "Stop after token"),
    ]

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        benchmark_file: Path = DEFAULT_BENCHMARK_FILE,
        results_file: Path = Path("benchmark_tui_results.jsonl"),
        gpu0_max_memory: str = DEFAULT_GPU0_MAX_MEMORY,
        gpu1_max_memory: str = DEFAULT_GPU1_MAX_MEMORY,
        autoload: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.benchmark_file = benchmark_file
        self.results_file = results_file
        self.gpu0_max_memory = gpu0_max_memory
        self.gpu1_max_memory = gpu1_max_memory
        self.autoload = autoload
        self.model = None
        self.tokenizer = None
        self.messages: list[dict] = []
        self.results: list[dict] = []
        self.busy = False
        self.stop_requested = False
        self.last_error_trace = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Starting…", id="status-bar")
        with TabbedContent(initial="chat"):
            with TabPane("Conversation", id="chat"):
                with Horizontal(classes="toolbar"):
                    yield Label("Mode")
                    yield Select(
                        [
                            ("Soft recurrent", "soft_recurrent"),
                            ("Baseline", "baseline"),
                            ("Hard argmax", "hard_argmax"),
                            ("Ordinary CoT", "ordinary_cot"),
                        ],
                        value="soft_recurrent",
                        allow_blank=False,
                        id="chat-mode",
                        classes="medium-select",
                    )
                    yield Label("Steps")
                    yield Input("8", type="integer", id="soft-steps", classes="small-input")
                    yield Label("Support")
                    yield Input("8", type="number", id="target-support", classes="small-input")
                    yield Label("Tokens")
                    yield Input("256", type="integer", id="max-tokens", classes="small-input")
                    yield Button("Clear", id="clear-chat")
                with Grid(id="chat-layout"):
                    with Vertical(id="chat-main"):
                        yield RichLog(id="chat-transcript", markup=True, wrap=True)
                        yield TextArea("", read_only=True, show_line_numbers=False, id="chat-live")
                        with Horizontal(id="chat-entry"):
                            yield Input(placeholder="Ask a follow-up…", id="chat-input")
                            yield Button("Send", id="send", variant="primary", disabled=True)
                    with Vertical(id="chat-side"):
                        yield Label("Soft-step diagnostics", classes="output-title")
                        yield RichLog(id="chat-trace", markup=True, wrap=True)
            with TabPane("Benchmark", id="benchmark"):
                with Horizontal(classes="toolbar"):
                    yield Label("Difficulty")
                    yield Select(
                        [("Challenge", "challenge"), ("Calibration", "calibration"), ("All", "all")],
                        value="challenge",
                        allow_blank=False,
                        id="benchmark-difficulty",
                        classes="medium-select",
                    )
                    yield Label("Steps")
                    yield Input("8", type="integer", id="benchmark-soft-steps", classes="small-input")
                    yield Label("Softmax")
                    yield Select(
                        [("Adaptive", "adaptive"), ("Fixed T", "fixed")],
                        value="adaptive",
                        allow_blank=False,
                        id="benchmark-softmax-mode",
                        classes="medium-select",
                    )
                    yield Label("Support")
                    yield Input("8", type="number", id="benchmark-support", classes="small-input")
                    yield Label("Fixed T")
                    yield Input("1.0", type="number", id="benchmark-temperature", classes="small-input")
                    yield Label("Top-k")
                    yield Input("64", type="integer", id="benchmark-top-k", classes="small-input")
                with Horizontal(classes="toolbar"):
                    yield Label("Min T")
                    yield Input("0.1", type="number", id="benchmark-min-temperature", classes="small-input")
                    yield Label("Max T")
                    yield Input("4.0", type="number", id="benchmark-max-temperature", classes="small-input")
                    yield Label("Entropy stop")
                    yield Input("0.75", type="number", id="benchmark-entropy-stop", classes="small-input")
                    yield Label("Tokens")
                    yield Input("512", type="integer", id="benchmark-max-tokens", classes="small-input")
                    yield Button("Run", id="run-benchmark", variant="primary", disabled=True)
                    yield Button("Stop", id="stop", variant="warning", disabled=True)
                    yield ProgressBar(total=48, show_eta=True, id="benchmark-progress")
                    yield Label("Ready", id="benchmark-status")
                yield Static(
                    "Steps controls latent duration; support = exp(entropy) controls mixture breadth. Negative entropy stop disables early exit.",
                    id="benchmark-help",
                )
                with Grid(id="benchmark-layout"):
                    yield DataTable(id="case-table", cursor_type="row", zebra_stripes=True)
                    yield DataTable(id="summary-table", cursor_type="row", zebra_stripes=True)
                    with Grid(id="output-grid"):
                        for mode in MODES:
                            with Vertical(classes="output-panel"):
                                yield Label(
                                    MODE_LABELS[mode],
                                    id=f"title-{mode}",
                                    classes="output-title",
                                )
                                yield Log(
                                    auto_scroll=True,
                                    max_lines=4000,
                                    id=f"output-{mode}",
                                    classes="output-text",
                                )
        yield Footer()

    def on_mount(self) -> None:
        case_table = self.query_one("#case-table", DataTable)
        for label, key in (
            ("Case", "case"),
            ("Category", "category"),
            ("Base", "baseline"),
            ("Hard", "hard_argmax"),
            ("Soft", "soft_recurrent"),
            ("CoT", "ordinary_cot"),
        ):
            case_table.add_column(label, key=key)
        summary = self.query_one("#summary-table", DataTable)
        for label, key in (
            ("Mode", "mode"),
            ("Correct", "correct"),
            ("Accuracy", "accuracy"),
            ("Δ baseline", "delta"),
            ("Tok/s", "throughput"),
            ("Avg latency", "latency"),
        ):
            summary.add_column(label, key=key)
        for mode in MODES:
            summary.add_row(MODE_LABELS[mode], "0/0", "—", "—", "—", "—", key=mode)
        self._sync_softmax_controls()
        if self.autoload:
            self.load_checkpoint()
        else:
            self._set_status("Model loading disabled for test")

    def _set_status(self, message: str) -> None:
        self.query_one("#status-bar", Static).update(message)

    def _sync_softmax_controls(self) -> None:
        adaptive = self.query_one("#benchmark-softmax-mode", Select).value == "adaptive"
        self.query_one("#benchmark-temperature", Input).disabled = adaptive
        for selector in (
            "#benchmark-support",
            "#benchmark-min-temperature",
            "#benchmark-max-temperature",
        ):
            self.query_one(selector, Input).disabled = not adaptive

    def _set_ready(self, ready: bool) -> None:
        self.busy = not ready
        unavailable = not ready or self.model is None
        self.query_one("#send", Button).disabled = unavailable
        self.query_one("#run-benchmark", Button).disabled = unavailable
        self.query_one("#stop", Button).disabled = ready or self.model is None

    def _load_succeeded(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self._set_ready(True)
        self._set_status(
            f"Ready · {self.model_path.name} · GPU0≤{self.gpu0_max_memory} GPU1≤{self.gpu1_max_memory}"
        )
        self.query_one("#chat-input", Input).focus()

    def _load_failed(self, message: str) -> None:
        self._set_ready(False)
        self._set_status(f"Load failed: {message}")
        self.notify(message, title="Model load failed", severity="error", timeout=10)

    def _generation_failed(self, message: str) -> None:
        self._set_ready(True)
        self._set_status(message)
        self.notify(message, title="Generation failed", severity="error", timeout=10)

    @work(thread=True, exclusive=True, group="model-load")
    def load_checkpoint(self) -> None:
        try:
            transformers_logging.disable_progress_bar()
            validate_cuda_environment()
            max_memory = build_max_memory_map(
                self.gpu0_max_memory, self.gpu1_max_memory
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            model = Qwen3_5ForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map="sequential",
                max_memory=max_memory,
                local_files_only=True,
            )
            model.eval()
        except (Exception, SystemExit) as error:
            self.last_error_trace = traceback.format_exc()
            self.call_from_thread(self._load_failed, f"{type(error).__name__}: {error}")
            return
        self.call_from_thread(self._load_succeeded, tokenizer, model)

    def _read_chat_settings(self) -> dict:
        return {
            "soft_steps": parse_positive_int(
                self.query_one("#soft-steps", Input).value, "Soft steps"
            ),
            "target_support": parse_positive_float(
                self.query_one("#target-support", Input).value, "Target support"
            ),
            "max_new_tokens": parse_positive_int(
                self.query_one("#max-tokens", Input).value, "Max tokens"
            ),
            "soft_temperature": None,
            "soft_top_k": 64,
            "entropy_stop": 0.75,
            "min_soft_temperature": 0.1,
            "max_soft_temperature": 4.0,
        }

    def _read_benchmark_settings(self) -> dict:
        adaptive = self.query_one("#benchmark-softmax-mode", Select).value == "adaptive"
        minimum = parse_positive_float(
            self.query_one("#benchmark-min-temperature", Input).value,
            "Minimum temperature",
        )
        maximum = parse_positive_float(
            self.query_one("#benchmark-max-temperature", Input).value,
            "Maximum temperature",
        )
        if minimum > maximum:
            raise ValueError("Minimum temperature must not exceed maximum temperature")
        entropy_stop = float(self.query_one("#benchmark-entropy-stop", Input).value)
        if entropy_stop >= 0 and not 0 < entropy_stop <= 1:
            raise ValueError("Entropy stop must be in (0, 1], or negative to disable")
        return {
            "soft_steps": parse_positive_int(
                self.query_one("#benchmark-soft-steps", Input).value,
                "Soft steps",
            ),
            "target_support": parse_positive_float(
                self.query_one("#benchmark-support", Input).value,
                "Target support",
            ),
            "max_new_tokens": parse_positive_int(
                self.query_one("#benchmark-max-tokens", Input).value,
                "Max tokens",
            ),
            "soft_temperature": None
            if adaptive
            else parse_positive_float(
                self.query_one("#benchmark-temperature", Input).value,
                "Fixed temperature",
            ),
            "soft_top_k": parse_nonnegative_int(
                self.query_one("#benchmark-top-k", Input).value,
                "Top-k",
            ),
            "entropy_stop": None if entropy_stop < 0 else entropy_stop,
            "min_soft_temperature": minimum,
            "max_soft_temperature": maximum,
        }

    def _update_text(self, selector: str, text: str) -> None:
        self.query_one(selector, TextArea).load_text(text)
        self.query_one(selector, TextArea).scroll_end(animate=False)

    def _append_trace(self, line: str) -> None:
        self.query_one("#chat-trace", RichLog).write(line)

    def _soft_trace(self, index, step, top: str) -> None:
        line = (
            f"[bold]{index:02d}[/] T={step.temperature:.3f} "
            f"support={step.effective_support:.1f} mass={step.retained_mass:.3f}\n{top}"
        )
        self.call_from_thread(self._append_trace, line)

    def _stream_to(self, selector: str) -> Callable[[str], None]:
        started = time.perf_counter()
        tokens = 0

        def stream(text: str) -> None:
            nonlocal tokens
            tokens += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            self.call_from_thread(self._update_text, selector, text)
            self.call_from_thread(
                self._set_status,
                f"Generating · {tokens} output tokens · {tokens / elapsed:.1f} tok/s",
            )

        return stream

    def _append_output(self, selector: str, text: str, reset: bool) -> None:
        output = self.query_one(selector, Log)
        if reset:
            output.clear()
        if text:
            output.write(text, scroll_end=True)

    def _update_output_title(self, mode: str, title: str) -> None:
        self.query_one(f"#title-{mode}", Label).update(title)

    def _stream_to_log(
        self,
        selector: str,
        mode: str,
        case_id: str,
    ) -> tuple[Callable[[str], None], dict]:
        previous = ""
        metrics = {"tokens": 0, "started": time.perf_counter(), "first_token_at": None}

        def stream(text: str) -> None:
            nonlocal previous
            metrics["tokens"] += 1
            if metrics["first_token_at"] is None:
                metrics["first_token_at"] = time.perf_counter()
            reset = not text.startswith(previous)
            delta = text if reset else text[len(previous):]
            previous = text
            elapsed = max(time.perf_counter() - metrics["started"], 1e-9)
            rate = metrics["tokens"] / elapsed
            self.call_from_thread(self._append_output, selector, delta, reset)
            self.call_from_thread(
                self._update_output_title,
                mode,
                f"{MODE_LABELS[mode]} · {case_id} · {metrics['tokens']} tok · {rate:.1f} tok/s",
            )

        return stream, metrics

    def _common_generation(self, settings: dict) -> dict:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": "",
            "max_new_tokens": settings["max_new_tokens"],
            "output_temperature": 0.0,
            "output_top_p": 0.95,
            "seed": 0,
            "should_stop": lambda: self.stop_requested,
        }

    def _generate_mode(
        self,
        mode: str,
        prompt: str,
        settings: dict,
        on_text: Callable[[str], None],
        messages: Sequence[dict] | None = None,
        on_soft_step: Callable | None = None,
    ) -> str:
        common = self._common_generation(settings)
        common["prompt"] = prompt
        common["messages"] = messages
        common["on_text"] = on_text
        if mode == "baseline":
            return soft_recurrent_answer(
                **common,
                soft_steps=0,
                soft_temperature=settings["soft_temperature"],
                soft_top_k=settings["soft_top_k"],
                entropy_stop=settings["entropy_stop"],
                target_support=settings["target_support"],
                min_soft_temperature=settings["min_soft_temperature"],
                max_soft_temperature=settings["max_soft_temperature"],
                show_trace=False,
            )[0]
        if mode == "hard_argmax":
            return hard_argmax_answer(
                **common,
                latent_steps=settings["soft_steps"],
            )[0]
        if mode == "ordinary_cot":
            return ordinary_cot_answer(
                **common,
                return_visible_only=messages is not None,
            )
        return soft_recurrent_answer(
            **common,
            soft_steps=settings["soft_steps"],
            soft_temperature=settings["soft_temperature"],
            soft_top_k=settings["soft_top_k"],
            entropy_stop=settings["entropy_stop"],
            target_support=settings["target_support"],
            min_soft_temperature=settings["min_soft_temperature"],
            max_soft_temperature=settings["max_soft_temperature"],
            show_trace=False,
            on_soft_step=on_soft_step,
        )[0]

    def _finish_chat(self, prompt: str, answer: str, stopped: bool) -> None:
        transcript = self.query_one("#chat-transcript", RichLog)
        if stopped:
            transcript.write("[yellow]Generation stopped; turn was not added to history.[/]")
        else:
            self.messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )
            entry = Text("Assistant\n", style="bold #e8b68f")
            entry.append(answer)
            transcript.write(entry)
        self._set_ready(True)
        self._set_status("Ready")
        self.query_one("#chat-input", Input).focus()

    @work(thread=True, exclusive=True, group="inference")
    def run_chat_turn(self, prompt: str, mode: str, settings: dict) -> None:
        messages = [*self.messages, {"role": "user", "content": prompt}]
        try:
            answer = self._generate_mode(
                mode,
                prompt,
                settings,
                self._stream_to("#chat-live"),
                messages=messages,
                on_soft_step=self._soft_trace,
            )
        except Exception as error:
            self.call_from_thread(
                self._generation_failed, f"Generation failed: {type(error).__name__}: {error}"
            )
            return
        self.call_from_thread(self._finish_chat, prompt, answer, self.stop_requested)

    def _start_chat(self) -> None:
        if self.busy or self.model is None:
            return
        prompt_input = self.query_one("#chat-input", Input)
        prompt = prompt_input.value.strip()
        if not prompt:
            return
        try:
            settings = self._read_chat_settings()
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        mode = str(self.query_one("#chat-mode", Select).value)
        prompt_input.value = ""
        self.query_one("#chat-live", TextArea).load_text("")
        self.query_one("#chat-trace", RichLog).clear()
        entry = Text("You\n", style="bold #7fb5d6")
        entry.append(prompt)
        self.query_one("#chat-transcript", RichLog).write(entry)
        self.stop_requested = False
        self._set_ready(False)
        self._set_status(f"Generating · {MODE_LABELS[mode]}")
        self.run_chat_turn(prompt, mode, settings)

    def _prepare_benchmark(self, cases: Sequence[dict]) -> None:
        table = self.query_one("#case-table", DataTable)
        table.clear()
        for case in cases:
            table.add_row(
                case["id"], case["category"].replace("_", " "), "·", "·", "·", "·", key=case["id"]
            )
        progress = self.query_one("#benchmark-progress", ProgressBar)
        progress.update(total=len(cases) * len(MODES), progress=0)
        self.query_one("#benchmark-status", Label).update(f"0/{len(cases) * 4}")
        for mode in MODES:
            self.query_one(f"#output-{mode}", Log).clear()

    def _mark_running(self, case: dict, mode: str) -> None:
        self.query_one("#case-table", DataTable).update_cell(
            case["id"], mode, "RUN", update_width=True
        )
        self.query_one("#benchmark-status", Label).update(
            f"{case['id']} · {MODE_LABELS[mode]}"
        )
        self.query_one(f"#title-{mode}", Label).update(
            f"{MODE_LABELS[mode]} · {case['id']}"
        )
        self.query_one(f"#output-{mode}", Log).clear()

    def _record_result(self, result: dict, completed: int, total: int) -> None:
        answer = result["normalized_answer"] or "—"
        marker = "✓" if result["correct"] else "✗"
        rate = float(result.get("tokens_per_second", 0.0))
        self.query_one("#case-table", DataTable).update_cell(
            result["case_id"],
            result["mode"],
            f"{marker} {answer} · {rate:.1f}t/s",
            update_width=True,
        )
        self.query_one("#benchmark-progress", ProgressBar).update(progress=completed)
        self.query_one("#benchmark-status", Label).update(f"{completed}/{total}")
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        table = self.query_one("#summary-table", DataTable)
        summaries = summarize_results(self.results)
        baseline_accuracy = next(row[3] for row in summaries if row[0] == "baseline")
        for mode, correct, total, accuracy, latency, throughput in summaries:
            table.update_cell(mode, "correct", f"{correct}/{total}")
            table.update_cell(mode, "accuracy", f"{accuracy:.1%}" if total else "—")
            delta = accuracy - baseline_accuracy
            table.update_cell(
                mode,
                "delta",
                f"{delta:+.1%}" if total and mode != "baseline" else "—",
            )
            table.update_cell(mode, "throughput", f"{throughput:.1f}" if total else "—")
            table.update_cell(mode, "latency", f"{latency / 1000:.1f}s" if total else "—")

    def _finish_benchmark(self, stopped: bool) -> None:
        self._set_ready(True)
        self._set_status("Benchmark stopped" if stopped else "Benchmark complete")
        self.query_one("#benchmark-status", Label).update(
            f"Saved {len(self.results)} results · {self.results_file}"
        )

    def _write_results(self) -> None:
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with self.results_file.open("w", encoding="utf-8") as handle:
            for result in self.results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    @work(thread=True, exclusive=True, group="inference")
    def run_benchmark_worker(self, cases: Sequence[dict], settings: dict) -> None:
        total = len(cases) * len(MODES)
        for case in cases:
            for mode in MODES:
                if self.stop_requested:
                    self.call_from_thread(self._finish_benchmark, True)
                    return
                self.call_from_thread(self._mark_running, case, mode)
                started = time.perf_counter()
                error = None
                latent_metrics = {"attempted": 0, "completed": 0, "entropy_stopped": False}
                stream, stream_metrics = self._stream_to_log(
                    f"#output-{mode}", mode, case["id"]
                )

                def soft_step(index, step, _top) -> None:
                    stopped = (
                        settings["entropy_stop"] is not None
                        and step.normalized_entropy >= settings["entropy_stop"]
                    )
                    latent_metrics["attempted"] = index
                    latent_metrics["entropy_stopped"] = stopped
                    if not stopped:
                        latent_metrics["completed"] = index
                    state = "entropy stop" if stopped else f"latent {index}/{settings['soft_steps']}"
                    self.call_from_thread(
                        self._update_output_title,
                        mode,
                        f"{MODE_LABELS[mode]} · {case['id']} · {state} · T={step.temperature:.2f} · S={step.effective_support:.1f}",
                    )

                try:
                    raw = self._generate_mode(
                        mode,
                        case["prompt"],
                        settings,
                        stream,
                        on_soft_step=soft_step if mode == "soft_recurrent" else None,
                    )
                except Exception as caught:
                    raw = ""
                    error = f"execution_error: {type(caught).__name__}: {caught}"
                if self.stop_requested and error is None:
                    error = "cancelled"
                scored = score_answer(raw, case["accepted_answers"])
                latency_ms = (time.perf_counter() - started) * 1000
                output_tokens = int(stream_metrics["tokens"])
                tokens_per_second = output_tokens / max(latency_ms / 1000, 1e-9)
                first_token_at = stream_metrics["first_token_at"]
                time_to_first_token_ms = (
                    (first_token_at - started) * 1000 if first_token_at is not None else None
                )
                result = {
                    "case_id": case["id"],
                    "category": case["category"],
                    "difficulty": case["difficulty"],
                    "mode": mode,
                    "config": {
                        **settings,
                        "model": str(self.model_path),
                        "gpu0_max_memory": self.gpu0_max_memory,
                        "gpu1_max_memory": self.gpu1_max_memory,
                    },
                    "raw_output": raw,
                    "extracted_answer": scored["answer"],
                    "normalized_answer": scored["normalized_answer"],
                    "correct": scored["correct"] if error is None else False,
                    "status": error or scored["status"],
                    "output_tokens": output_tokens,
                    "tokens_per_second": round(tokens_per_second, 3),
                    "time_to_first_token_ms": round(time_to_first_token_ms, 3)
                    if time_to_first_token_ms is not None
                    else None,
                    "latent_steps_completed": latent_metrics["completed"]
                    if mode == "soft_recurrent"
                    else None,
                    "latent_entropy_stopped": latent_metrics["entropy_stopped"]
                    if mode == "soft_recurrent"
                    else None,
                    "latency_ms": round(latency_ms, 3),
                }
                self.results.append(result)
                self._write_results()
                self.call_from_thread(self._record_result, result, len(self.results), total)
        self.call_from_thread(self._finish_benchmark, False)

    def action_start_benchmark(self) -> None:
        if self.busy or self.model is None:
            return
        try:
            settings = self._read_benchmark_settings()
            cases = load_benchmark_cases(self.benchmark_file)
            difficulty = str(self.query_one("#benchmark-difficulty", Select).value)
            if difficulty != "all":
                cases = select_benchmark_cases(cases, difficulties=[difficulty])
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        self.results = []
        self.stop_requested = False
        self._prepare_benchmark(cases)
        self._set_ready(False)
        self._set_status(f"Benchmarking {len(cases)} cases")
        self.run_benchmark_worker(cases, settings)

    def action_request_stop(self) -> None:
        if self.busy and self.model is not None:
            self.stop_requested = True
            self._set_status("Stopping after the current token…")

    def action_clear_chat(self) -> None:
        if self.busy:
            return
        self.messages.clear()
        self.query_one("#chat-transcript", RichLog).clear()
        self.query_one("#chat-live", TextArea).load_text("")
        self.query_one("#chat-trace", RichLog).clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self._start_chat()
        elif event.button.id == "clear-chat":
            self.action_clear_chat()
        elif event.button.id == "run-benchmark":
            self.action_start_benchmark()
        elif event.button.id == "stop":
            self.action_request_stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._start_chat()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "benchmark-softmax-mode":
            self._sync_softmax_controls()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "case-table":
            return
        case_id = str(event.row_key.value)
        selected = [result for result in self.results if result["case_id"] == case_id]
        for result in selected:
            mode = result["mode"]
            self.query_one(f"#title-{mode}", Label).update(
                f"{MODE_LABELS[mode]} · {case_id}"
            )
            output = self.query_one(f"#output-{mode}", Log)
            output.clear()
            output.write(result["raw_output"], scroll_end=True)
        if selected:
            self.query_one("#benchmark-status", Label).update(
                f"Viewing {case_id} · {len(selected)}/4 conditions"
            )


def parse_tui_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive neuralese test harness")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--results-file", type=Path, default=Path("benchmark_tui_results.jsonl"))
    parser.add_argument("--gpu0-max-memory", default=DEFAULT_GPU0_MAX_MEMORY)
    parser.add_argument("--gpu1-max-memory", default=DEFAULT_GPU1_MAX_MEMORY)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_tui_args()
    NeuraleseTUI(
        model_path=args.model,
        benchmark_file=args.benchmark_file,
        results_file=args.results_file,
        gpu0_max_memory=args.gpu0_max_memory,
        gpu1_max_memory=args.gpu1_max_memory,
    ).run()


if __name__ == "__main__":
    main()
