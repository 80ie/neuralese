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
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
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
    hidden_recurrent_answer,
    interleaved_recurrent_answer,
    load_benchmark_cases,
    ordinary_cot_answer,
    score_answer,
    select_benchmark_cases,
    soft_recurrent_answer,
    validate_cuda_environment,
)


MODES = ("baseline", "hard_argmax", "soft_recurrent", "hidden_recurrent", "ordinary_cot", "interleaved_recurrent")
MODE_LABELS = {
    "baseline": "Baseline",
    "hard_argmax": "Hard argmax",
    "soft_recurrent": "Soft recurrent",
    "hidden_recurrent": "Hidden recurrent",
    "ordinary_cot": "Ordinary CoT",
    "interleaved_recurrent": "Interleaved recurrent",
}
MODE_SHORT_LABELS = {
    "baseline": "Baseline",
    "hard_argmax": "Hard argmax",
    "soft_recurrent": "Soft recur.",
    "hidden_recurrent": "Hidden recur.",
    "ordinary_cot": "Ordinary CoT",
    "interleaved_recurrent": "Interleaved",
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


class OutputText(TextArea):
    """Wrapped output that follows the tail only while the reader is at the tail."""

    def __init__(self, **kwargs) -> None:
        super().__init__(read_only=True, show_line_numbers=False, soft_wrap=True, **kwargs)
        self.cursor_blink = False
        # This is an output viewer: an editor cursor must not pull the viewport
        # away from the reader or compete with tail-following during insertion.
        self.show_cursor = False

    def on_resize(self) -> None:
        # Textual rewraps the document on resize, but if its wrapped height is
        # unchanged the compositor can reuse stale, cropped lines.
        self.refresh()

    def show_text(self, text: str, *, changed_selection: bool = False) -> None:
        current = self.text
        if text == current and not changed_selection:
            return
        following = self.scroll_y >= self.max_scroll_y - 1
        if changed_selection:
            self.load_text(text)
            self.scroll_home(animate=False, immediate=True)
            return

        selection = self.selection
        if text.startswith(current):
            self.insert(text[len(current):], self.document.end)
        else:
            # A tokenizer may revise its last decoded characters. Replace only
            # the changed suffix, retaining the document and its wrapped prefix.
            prefix = 0
            for old, new in zip(current, text):
                if old != new:
                    break
                prefix += 1
            self.replace(text[prefix:], self.document.get_location_from_index(prefix), self.document.end)
        self.selection = selection
        self.history.clear()
        if following:
            self.scroll_end(animate=False)


class NeuraleseTUI(App):
    TITLE = "Neuralese Harness"
    SUB_TITLE = "Soft-token recurrence lab"
    CSS = """
    Screen { background: $background; color: $text-primary; }
    Header { background: $panel; }
    #status-bar { height: 1; padding: 0 1; background: $surface; color: $text-muted; }
    TabbedContent { height: 1fr; }
    .toolbar { height: 3; padding: 0 1; align-vertical: middle; overflow-x: auto; overflow-y: hidden; }
    .toolbar Label { width: auto; margin: 0 1 0 0; }
    /* Keep controls at their useful size. Flex shrinking made the inputs
       collapse to only a couple of visible characters in the benchmark bar. */
    .small-input { width: 14; min-width: 14; padding: 0 1; margin-right: 1; }
    .medium-select { width: 18; min-width: 18; margin-right: 1; }
    Button { margin-right: 1; min-width: 10; }
    #chat-layout { height: 1fr; grid-size: 2 1; grid-columns: 2fr 1fr; grid-gutter: 1; }
    #chat-main, #chat-side { height: 1fr; }
    #chat-transcript { height: 1fr; border: round $primary; padding: 0 1; }
    #chat-live { height: 12; border: round $accent; }
    #chat-trace { height: 1fr; border: round $primary; }
    #chat-entry { height: 3; }
    #chat-input { width: 1fr; }
    #benchmark { padding: 0 1; }
    #benchmark-controls { height: 3; align-vertical: middle; }
    #benchmark-controls Button { min-width: 8; }
    #benchmark-config { width: 1fr; height: 1; color: $text-muted; }
    #run-progress { height: 1; margin-bottom: 1; }
    #benchmark-progress { width: 22; margin-right: 2; }
    #benchmark-status { width: 1fr; height: 1; }
    #benchmark-dashboard { height: 1fr; }
    #benchmark-overview { height: 1fr; min-height: 9; max-height: 14; grid-size: 2 1; grid-columns: 1fr 1fr; grid-gutter: 1; }
    #case-table, #summary-table { height: 1fr; border: round $panel; border-title-color: $text-muted; background: $surface; }
    #case-table:focus, #summary-table:focus { border: round $accent; }
    #benchmark DataTable > .datatable--header { background: $panel; color: $text-primary; text-style: bold; }
    #benchmark DataTable > .datatable--odd-row { background: $boost; }
    #benchmark DataTable > .datatable--even-row { background: $surface; }
    #benchmark DataTable > .datatable--cursor { background: $primary-muted; color: $text-primary; }
    #overview-switch, #output-switch { height: 1; display: none; }
    #show-output { display: none; }
    .text-button { height: 1; min-width: 8; width: auto; border: none; padding: 0 1; margin: 0 1 0 0; background: $panel; }
    .text-button.-active { background: $primary-muted; color: $text-primary; text-style: bold; }
    #case-context { height: 2; margin-top: 1; }
    #viewing-case { width: 1fr; height: 1; text-style: bold; }
    #follow-live { width: 20; }
    #case-answer { height: 1; color: $text-muted; }
    #compare-controls { height: 3; align-vertical: middle; }
    #compare-controls Label { width: auto; margin-right: 1; }
    #comparison-mode { width: 29; margin-right: 2; }
    #output-grid { height: 2fr; grid-size: 2 1; grid-columns: 1fr 1fr; grid-gutter: 1; }
    .output-panel { height: 1fr; border: round $panel; background: $surface; }
    .pane-heading { height: 1; background: $panel; }
    .pane-heading Label { width: 1fr; padding-left: 1; text-style: bold; }
    .pane-metrics { height: 3; padding: 0 1; color: $text-muted; }
    OutputText { height: 1fr; border: none; background: $surface; padding: 0 1; }
    OutputText:focus { border: none; }
    #benchmark-saved { height: 1; color: $text-disabled; }
    #benchmark-settings { display: none; height: 1fr; border: round $panel; padding: 0 2; }
    .settings-heading { height: 2; padding-top: 1; color: $accent; text-style: bold; }
    .settings-help { height: auto; color: $text-muted; margin-bottom: 1; max-width: 85; }
    .settings-row { height: 3; align-vertical: middle; }
    .settings-row Label { width: 27; }
    .settings-row Input, .settings-row Select { width: 24; }
    #settings-error { height: auto; color: $error; }
    #benchmark-prompt { display: none; height: 1fr; border: round $panel; padding: 1; }
    #prompt-title { height: 1; text-style: bold; }
    #prompt-text { height: 1fr; margin-bottom: 1; }
    #benchmark.narrow #benchmark-overview { grid-size: 1 1; grid-columns: 1fr; }
    #benchmark.narrow #overview-switch, #benchmark.narrow #output-switch { display: block; }
    #benchmark.narrow #output-grid { grid-size: 1 1; grid-columns: 1fr; }
    #benchmark.short #benchmark-overview { height: 1fr; min-height: 0; max-height: 100%; }
    #benchmark.short #show-output { display: block; }
    #benchmark.short #benchmark-controls { height: 1; }
    #benchmark.short #benchmark-controls Button { height: 1; border: none; padding: 0 1; }
    #benchmark.short .pane-metrics { height: 2; }
    #benchmark.short #benchmark-saved { display: none; }
    #benchmark.short #run-progress { margin-bottom: 0; }
    #benchmark.short #case-context { margin-top: 0; }
    #benchmark.expanded #benchmark-overview, #benchmark.expanded #overview-switch { display: none; }
    #benchmark.expanded #output-grid { grid-size: 1 1; grid-columns: 1fr; }
    .output-title { height: 1; padding: 0 1; background: $accent-muted; color: $accent; }
    .output-text { height: 1fr; }
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
        self.main_screen = None
        self.benchmark_cases: dict[str, dict] = {}
        self.output_buffers: dict[tuple[str, str], str] = {}
        self.output_diagnostics: dict[tuple[str, str], str] = {}
        self.viewed_case: str | None = None
        self.running_case: str | None = None
        self.running_mode: str | None = None
        self.follow_live = True
        self.comparison_mode = "soft_recurrent"
        self.overview_page = "cases"
        self.compact_page = "output"
        self.output_page = "comparison"
        self.expanded_pane: str | None = None
        self.settings_open = False
        self.prompt_open = False
        self.benchmark_started: float | None = None
        self.benchmark_finished: float | None = None
        self.benchmark_state = "Ready"
        self.rendered_output_keys: dict[str, tuple[str | None, str]] = {}

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
                            ("Hidden recurrent", "hidden_recurrent"),
                            ("Baseline", "baseline"),
                            ("Hard argmax", "hard_argmax"),
                            ("Ordinary CoT", "ordinary_cot"),
                            ("Interleaved", "interleaved_recurrent"),
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
                with Horizontal(classes="toolbar"):
                    yield Label("Hidden/token")
                    yield Input("2", type="integer", id="interleaved-hidden-steps", classes="small-input")
                    yield Label("Thinking tokens")
                    yield Input("64", type="integer", id="interleaved-thinking-tokens", classes="small-input")
                    yield Label("Interleaved settings apply only to Interleaved mode", id="chat-interleaved-help")
                with Grid(id="chat-layout"):
                    with Vertical(id="chat-main"):
                        yield RichLog(id="chat-transcript", markup=True, wrap=True)
                        yield TextArea("", read_only=True, show_line_numbers=False, id="chat-live")
                        with Horizontal(id="chat-entry"):
                            yield Input(placeholder="Ask a follow-up…", id="chat-input")
                            yield Button("Send", id="send", variant="primary", disabled=True)
                    with Vertical(id="chat-side"):
                        yield Label("Recurrence diagnostics", classes="output-title")
                        yield RichLog(id="chat-trace", markup=True, wrap=True)
            with TabPane("Benchmark", id="benchmark"):
                with Horizontal(id="benchmark-controls"):
                    yield Button("Run", id="run-benchmark", variant="primary", disabled=True)
                    yield Button("Stop", id="stop", variant="warning", disabled=True)
                    yield Button("Settings", id="settings")
                    yield Static("Challenge · 6 modes · 512 tokens", id="benchmark-config")
                with Horizontal(id="run-progress"):
                    yield ProgressBar(total=1, show_eta=False, show_percentage=False, id="benchmark-progress")
                    yield Static("Ready", id="benchmark-status", markup=False)
                with Vertical(id="benchmark-dashboard"):
                    with Horizontal(id="overview-switch"):
                        yield Button("Cases", id="show-cases", classes="text-button -active")
                        yield Button("Summary", id="show-summary", classes="text-button")
                        yield Button("Output", id="show-output", classes="text-button")
                    with Grid(id="benchmark-overview"):
                        yield DataTable(id="case-table", cursor_type="row", zebra_stripes=True)
                        yield DataTable(id="summary-table", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="case-context"):
                        with Horizontal():
                            yield Static("Select a case to inspect its output", id="viewing-case", markup=False)
                            yield Button("Follow live: on", id="follow-live", classes="text-button -active")
                        yield Static("Expected: —", id="case-answer", markup=False)
                    with Horizontal(id="compare-controls"):
                        yield Label("Baseline vs")
                        yield Select(
                            [(MODE_LABELS[mode], mode) for mode in MODES if mode != "baseline"],
                            value="soft_recurrent", allow_blank=False, id="comparison-mode",
                        )
                        yield Button("Full prompt", id="show-prompt", classes="text-button", disabled=True)
                    with Horizontal(id="output-switch"):
                        yield Button("Baseline", id="show-baseline", classes="text-button")
                        yield Button("Comparison", id="show-comparison", classes="text-button -active")
                    with Grid(id="output-grid"):
                        for pane in ("baseline", "comparison"):
                            with Vertical(id=f"pane-{pane}", classes="output-panel"):
                                with Horizontal(classes="pane-heading"):
                                    yield Label("", id=f"title-{pane}")
                                    yield Button("Expand", id=f"expand-{pane}", classes="text-button")
                                yield Static("", id=f"metrics-{pane}", classes="pane-metrics", markup=False)
                                yield OutputText(id=f"output-{pane}")
                    yield Static("Results are saved after each condition.", id="benchmark-saved", markup=False)
                with VerticalScroll(id="benchmark-settings"):
                    yield Label("Dataset and budget", classes="settings-heading")
                    with Horizontal(classes="settings-row"):
                        yield Label("Difficulty")
                        yield Select(
                            [("Challenge", "challenge"), ("Calibration", "calibration"), ("All", "all")],
                            value="challenge", allow_blank=False, id="benchmark-difficulty",
                        )
                    yield from self._setting_input("Output token budget", "benchmark-max-tokens", "512", "integer")
                    yield Label("Recurrence", classes="settings-heading")
                    yield Static("Latent steps apply to hard argmax, soft, and hidden recurrence.", classes="settings-help")
                    yield from self._setting_input("Latent steps", "benchmark-soft-steps", "8", "integer")
                    yield Label("Soft recurrence", classes="settings-heading")
                    with Horizontal(classes="settings-row"):
                        yield Label("Softmax")
                        yield Select(
                            [("Adaptive", "adaptive"), ("Fixed temperature", "fixed")],
                            value="adaptive", allow_blank=False, id="benchmark-softmax-mode",
                        )
                    yield from self._setting_input("Target support", "benchmark-support", "8")
                    yield from self._setting_input("Minimum temperature", "benchmark-min-temperature", "0.1")
                    yield from self._setting_input("Maximum temperature", "benchmark-max-temperature", "4.0")
                    yield from self._setting_input("Fixed temperature", "benchmark-temperature", "1.0")
                    yield from self._setting_input("Top-k", "benchmark-top-k", "64", "integer")
                    yield Static("Support is effective distribution size. Top-k limits candidates; 0 uses the full vocabulary.", classes="settings-help")
                    with Horizontal(classes="settings-row"):
                        yield Label("Entropy stopping")
                        yield Switch(True, id="benchmark-entropy-enabled")
                    yield from self._setting_input("Entropy threshold", "benchmark-entropy-stop", "0.75")
                    yield Static("Stop soft recurrence when normalized entropy reaches this threshold (0–1).", classes="settings-help")
                    yield Label("Interleaved recurrence", classes="settings-heading")
                    yield from self._setting_input("Hidden steps per token", "benchmark-interleaved-hidden-steps", "2", "integer")
                    yield from self._setting_input("Thinking-token budget", "benchmark-interleaved-thinking-tokens", "64", "integer")
                    yield Static("Hidden steps are inserted between real thinking tokens. These settings apply only to Interleaved.", classes="settings-help")
                    yield Static("", id="settings-error", markup=False)
                    yield Button("Done", id="settings-done", variant="primary")
                with Vertical(id="benchmark-prompt"):
                    yield Static("", id="prompt-title", markup=False)
                    yield TextArea("", read_only=True, show_line_numbers=False, id="prompt-text")
                    yield Button("Close prompt", id="close-prompt")
        yield Footer()

    def _setting_input(self, label: str, widget_id: str, value: str, kind: str = "number") -> ComposeResult:
        with Horizontal(classes="settings-row", id=f"row-{widget_id}"):
            yield Label(label)
            yield Input(value, type=kind, id=widget_id)

    def on_mount(self) -> None:
        # Keep worker updates on the dashboard even while Textual's command
        # palette (or another screen) is on top of it.
        self.main_screen = self.screen
        case_table = self.query_one("#case-table", DataTable)
        case_table.border_title = "Cases · ✓ pass  ✗ fail  ▶ running  · pending  ■ stopped  ! error"
        case_table.fixed_columns = 1
        for label, key in (
            ("Case", "case"),
            ("Base", "baseline"),
            ("Hard", "hard_argmax"),
            ("Soft", "soft_recurrent"),
            ("Hid", "hidden_recurrent"),
            ("CoT", "ordinary_cot"),
            ("IL", "interleaved_recurrent"),
        ):
            case_table.add_column(label, key=key, width=16 if key == "case" else 4)
        summary = self.query_one("#summary-table", DataTable)
        summary.border_title = "Mode summary"
        for label, key, width in (
            ("Mode", "mode", 13),
            ("Correct", "correct", 7),
            ("Acc", "accuracy", 6),
            ("Δ base", "delta", 7),
            ("Tok/s", "throughput", 6),
            ("Avg s", "latency", 6),
        ):
            summary.add_column(label, key=key, width=width)
        for mode in MODES:
            summary.add_row(MODE_SHORT_LABELS[mode], "0/0", "—", "—", "—", "—", key=mode)
        self._sync_softmax_controls()
        self._refresh_config_summary()
        self._refresh_inspector()
        self._apply_benchmark_layout()
        self.benchmark_timer = self.set_interval(1, self._refresh_benchmark_progress, pause=True)
        if self.autoload:
            self.load_checkpoint()
        else:
            self._set_status("Model loading disabled for test")

    def _set_status(self, message: str) -> None:
        self.main_screen.query_one("#status-bar", Static).update(message)

    def _sync_softmax_controls(self) -> None:
        adaptive = self.query_one("#benchmark-softmax-mode", Select).value == "adaptive"
        self.query_one("#benchmark-temperature", Input).disabled = adaptive
        self.query_one("#row-benchmark-temperature").display = not adaptive
        for selector in (
            "#benchmark-support",
            "#benchmark-min-temperature",
            "#benchmark-max-temperature",
        ):
            self.query_one(selector, Input).disabled = not adaptive
            self.query_one(f"#row-{selector[1:]}").display = adaptive

    def on_resize(self, event: events.Resize) -> None:
        if self.main_screen is not None:
            self._apply_benchmark_layout(event.size.width, event.size.height)

    def _apply_benchmark_layout(self, width: int | None = None, height: int | None = None) -> None:
        short = (height or self.size.height) < 34
        narrow = (width or self.size.width) < 140 or short
        benchmark = self.main_screen.query_one("#benchmark")
        benchmark.set_class(narrow, "narrow")
        benchmark.set_class(short, "short")
        benchmark.set_class(self.expanded_pane is not None, "expanded")
        show_output = not short or self.compact_page == "output" or self.expanded_pane is not None
        benchmark.query_one("#benchmark-overview").display = (
            self.expanded_pane is None and (not short or not show_output)
        )
        for selector in ("#case-context", "#compare-controls"):
            benchmark.query_one(selector).display = show_output and self.expanded_pane is None
        benchmark.query_one("#output-grid").display = show_output
        benchmark.query_one("#output-switch").display = narrow and show_output
        benchmark.query_one("#case-table").display = not narrow or self.overview_page == "cases"
        benchmark.query_one("#summary-table").display = not narrow or self.overview_page == "summary"
        for pane in ("baseline", "comparison"):
            visible = (
                self.expanded_pane == pane if self.expanded_pane
                else not narrow or self.output_page == pane
            )
            benchmark.query_one(f"#pane-{pane}").display = visible
            benchmark.query_one(f"#expand-{pane}", Button).label = "Restore" if self.expanded_pane == pane else "Expand"
            benchmark.query_one(f"#show-{pane}").set_class(self.output_page == pane, "-active")
        for page in ("cases", "summary"):
            benchmark.query_one(f"#show-{page}").set_class(self.overview_page == page and (not short or not show_output), "-active")
        benchmark.query_one("#show-output").set_class(short and show_output, "-active")

    def _refresh_config_summary(self) -> None:
        difficulty = str(self.query_one("#benchmark-difficulty", Select).value).title()
        tokens = self.query_one("#benchmark-max-tokens", Input).value
        self.query_one("#benchmark-config", Static).update(f"{difficulty} · 6 modes · {tokens} tokens")

    def _toggle_settings(self) -> None:
        if self.prompt_open:
            self._close_prompt()
        if self.settings_open:
            try:
                self._read_benchmark_settings()
            except ValueError as error:
                self.query_one("#settings-error", Static).update(str(error))
                self.query_one("#settings-error").scroll_visible()
                return
            self.query_one("#settings-error", Static).update("")
            self._refresh_config_summary()
        self.settings_open = not self.settings_open
        self.query_one("#benchmark-settings").display = self.settings_open
        self.query_one("#benchmark-dashboard").display = not self.settings_open
        self.query_one("#settings", Button).label = "Done" if self.settings_open else "Settings"
        if self.settings_open:
            self.query_one("#benchmark-difficulty").focus()
        else:
            self.query_one("#case-table").focus()

    def _refresh_benchmark_progress(self) -> None:
        if self.benchmark_started is None or not self.is_running:
            return
        elapsed = int((self.benchmark_finished or time.perf_counter()) - self.benchmark_started)
        total = len(self.benchmark_cases) * len(MODES)
        state = self.benchmark_state
        if state == "Running" and self.running_case:
            labels = MODE_SHORT_LABELS if self.size.width < 100 else MODE_LABELS
            mode = labels.get(self.running_mode, "")
            state = f"{self.running_case} / {mode}"
        self.main_screen.query_one("#benchmark-status", Static).update(
            f"{state} · {len(self.results)}/{total} · {elapsed // 60}:{elapsed % 60:02d}"
        )

    def _show_prompt(self) -> None:
        case = self.benchmark_cases.get(self.viewed_case)
        if case is None:
            return
        self.follow_live = False
        self._refresh_inspector()
        self.prompt_open = True
        self.query_one("#prompt-title", Static).update(f"{case['id']} · {case['category'].replace('_', ' ')}")
        self.query_one("#prompt-text", TextArea).load_text(
            case.get("prompt", "") + "\n\nAccepted answers: " + ", ".join(case.get("accepted_answers", []))
        )
        self.query_one("#benchmark-dashboard").display = False
        self.query_one("#benchmark-prompt").display = True
        self.query_one("#prompt-text").focus()

    def _close_prompt(self) -> None:
        self.prompt_open = False
        self.query_one("#benchmark-prompt").display = False
        self.query_one("#benchmark-dashboard").display = True
        self.query_one("#show-prompt").focus()

    def _result_for(self, case_id: str | None, mode: str) -> dict | None:
        return next((result for result in reversed(self.results)
                     if result["case_id"] == case_id and result["mode"] == mode), None)

    def _theme_color(self, name: str) -> str:
        """Return a Rich-compatible color from the currently active theme."""
        return str(self.theme_variables[name])

    def _result_state(self, result: dict) -> tuple[str, str, str]:
        status = result.get("status", "")
        if status == "cancelled":
            return "■", "STOPPED", self._theme_color("warning")
        if status.startswith("execution_error"):
            return "!", "ERROR", self._theme_color("error")
        if result["correct"]:
            return "✓", "PASS", self._theme_color("success")
        return "✗", "FAIL", self._theme_color("error")

    def _refresh_inspector(self) -> None:
        screen = self.main_screen
        case = self.benchmark_cases.get(self.viewed_case)
        screen.query_one("#viewing-case", Static).update(
            f"Viewing {case['id']} · {case['category'].replace('_', ' ')}" if case
            else "Select a case to inspect its output"
        )
        screen.query_one("#case-answer", Static).update(
            "Expected: " + ", ".join(case.get("accepted_answers", [])) if case else "Expected: —"
        )
        screen.query_one("#show-prompt", Button).disabled = case is None
        follow = screen.query_one("#follow-live", Button)
        follow.label = f"Follow live: {'on' if self.follow_live else 'off'}"
        follow.set_class(self.follow_live, "-active")
        for pane, mode in (("baseline", "baseline"), ("comparison", self.comparison_mode)):
            key = (self.viewed_case, mode)
            result = self._result_for(*key)
            active = key == (self.running_case, self.running_mode) and self.benchmark_state in ("Running", "Stopping")
            if result:
                marker, state, color = self._result_state(result)
                answer = result.get("extracted_answer") or result.get("normalized_answer") or "—"
                metrics = (
                    f"Answer: {answer}\n{result.get('output_tokens', 0)} tokens · "
                    f"{float(result.get('tokens_per_second', 0)):.1f} tok/s · "
                    f"{float(result.get('latency_ms', 0)) / 1000:.1f}s"
                )
                if state in ("ERROR", "STOPPED"):
                    metrics += f"\n{result.get('status', '')}"
                elif mode == "interleaved_recurrent":
                    metrics += f"\nThinking: {result.get('thinking_tokens_completed', '—')} · Hidden: {result.get('latent_steps_completed', '—')}"
                elif mode in ("soft_recurrent", "hidden_recurrent"):
                    metrics += f"\nLatent steps: {result.get('latent_steps_completed', '—')}"
                    if result.get("latent_entropy_stopped"):
                        metrics += " · entropy stop"
            else:
                marker, state, color = (
                    ("▶", "RUNNING", self._theme_color("primary"))
                    if active
                    else ("·", "PENDING", self._theme_color("text-muted"))
                )
                metrics = self.output_diagnostics.get(key, "Waiting for this condition." if case else "Run a benchmark to see results.")
            screen.query_one(f"#title-{pane}", Label).update(Text(f"{MODE_LABELS[mode]} · {marker} {state}", style=color))
            screen.query_one(f"#metrics-{pane}", Static).update(metrics)
            text = self.output_buffers.get(key, result.get("raw_output", "") if result else "")
            output = screen.query_one(f"#output-{pane}", OutputText)
            changed = self.rendered_output_keys.get(pane) != key
            output.show_text(text, changed_selection=changed)
            self.rendered_output_keys[pane] = key

    def _follow_running(self) -> None:
        if self.running_case:
            self.viewed_case = self.running_case
            if self.running_mode and self.running_mode != "baseline":
                self.comparison_mode = self.running_mode
                self.main_screen.query_one("#comparison-mode", Select).value = self.comparison_mode
            self.output_page = "baseline" if self.running_mode == "baseline" else "comparison"
            table = self.main_screen.query_one("#case-table", DataTable)
            table.move_cursor(row=table.get_row_index(self.running_case))
        self._refresh_inspector()
        self._apply_benchmark_layout()

    def _set_ready(self, ready: bool) -> None:
        self.busy = not ready
        unavailable = not ready or self.model is None
        self.main_screen.query_one("#send", Button).disabled = unavailable
        self.main_screen.query_one("#run-benchmark", Button).disabled = unavailable
        self.main_screen.query_one("#stop", Button).disabled = ready or self.model is None
        self.main_screen.query_one("#settings", Button).disabled = not ready and self.benchmark_state in ("Running", "Stopping")

    def _load_succeeded(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self._set_ready(True)
        self._set_status(
            f"Ready · {self.model_path.name} · GPU0≤{self.gpu0_max_memory} GPU1≤{self.gpu1_max_memory}"
        )
        self.main_screen.query_one("#chat-input", Input).focus()

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
            "hidden_steps_per_token": parse_nonnegative_int(
                self.query_one("#interleaved-hidden-steps", Input).value,
                "Hidden steps per token",
            ),
            "max_thinking_tokens": parse_positive_int(
                self.query_one("#interleaved-thinking-tokens", Input).value,
                "Max thinking tokens",
            ),
        }

    def _read_benchmark_settings(self) -> dict:
        adaptive = self.query_one("#benchmark-softmax-mode", Select).value == "adaptive"
        minimum = parse_positive_float(
            self.query_one("#benchmark-min-temperature", Input).value,
            "Minimum temperature",
        ) if adaptive else 0.1
        maximum = parse_positive_float(
            self.query_one("#benchmark-max-temperature", Input).value,
            "Maximum temperature",
        ) if adaptive else 4.0
        if minimum > maximum:
            raise ValueError("Minimum temperature must not exceed maximum temperature")
        entropy_stop = None
        if self.query_one("#benchmark-entropy-enabled", Switch).value:
            try:
                entropy_stop = float(self.query_one("#benchmark-entropy-stop", Input).value)
            except ValueError as error:
                raise ValueError("Entropy threshold must be a number") from error
            if not 0 < entropy_stop <= 1:
                raise ValueError("Entropy threshold must be in (0, 1]; use the switch to disable it")
        return {
            "soft_steps": parse_positive_int(
                self.query_one("#benchmark-soft-steps", Input).value,
                "Soft steps",
            ),
            "target_support": parse_positive_float(
                self.query_one("#benchmark-support", Input).value,
                "Target support",
            ) if adaptive else 8.0,
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
            "entropy_stop": entropy_stop,
            "min_soft_temperature": minimum,
            "max_soft_temperature": maximum,
            "hidden_steps_per_token": parse_nonnegative_int(
                self.query_one("#benchmark-interleaved-hidden-steps", Input).value,
                "Hidden steps per token",
            ),
            "max_thinking_tokens": parse_positive_int(
                self.query_one("#benchmark-interleaved-thinking-tokens", Input).value,
                "Max thinking tokens",
            ),
        }

    def _update_text(self, selector: str, text: str) -> None:
        self.main_screen.query_one(selector, TextArea).load_text(text)
        self.main_screen.query_one(selector, TextArea).scroll_end(animate=False)

    def _append_trace(self, line: str) -> None:
        self.main_screen.query_one("#chat-trace", RichLog).write(line)

    def _soft_trace(self, index, step, top: str) -> None:
        line = (
            f"[bold]{index:02d}[/] T={step.temperature:.3f} "
            f"support={step.effective_support:.1f} mass={step.retained_mass:.3f}\n{top}"
        )
        self.call_from_thread(self._append_trace, line)

    def _hidden_trace(self, index, rms: float) -> None:
        self.call_from_thread(self._append_trace, f"[bold]{index:02d}[/] hidden latent · RMS={rms:.4f}")

    def _interleaved_trace(self, index: int, text: str, hidden_steps: int, rms: float) -> None:
        self.call_from_thread(
            self._append_trace,
            f"[bold]{index:02d}[/] {text!r} · hidden gap={hidden_steps} · RMS={rms:.4f}",
        )

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

    def _append_output(self, case_id: str, mode: str, text: str, reset: bool) -> None:
        key = (case_id, mode)
        self.output_buffers[key] = text if reset else self.output_buffers.get(key, "") + text
        if case_id == self.viewed_case:
            self._refresh_inspector()

    def _update_output_title(self, mode: str, title: str, case_id: str | None = None) -> None:
        case_id = case_id or self.running_case
        if case_id is not None:
            self.output_diagnostics[(case_id, mode)] = title
        if case_id == self.viewed_case:
            self._refresh_inspector()

    def _stream_benchmark(
        self,
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
            self.call_from_thread(self._append_output, case_id, mode, delta, reset)
            self.call_from_thread(
                self._update_output_title,
                mode,
                f"Generating · {metrics['tokens']} tokens · {rate:.1f} tok/s",
                case_id,
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
        on_hidden_step: Callable | None = None,
        on_hidden_complete: Callable[[int], None] | None = None,
        on_interleaved_token: Callable | None = None,
        on_interleaved_complete: Callable[[int, int, bool], None] | None = None,
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
        if mode == "hidden_recurrent":
            answer, completed_steps = hidden_recurrent_answer(
                **common,
                latent_steps=settings["soft_steps"],
                use_thinking_scaffold=True,
                on_hidden_step=on_hidden_step,
            )
            if on_hidden_complete is not None:
                on_hidden_complete(completed_steps)
            return answer
        if mode == "interleaved_recurrent":
            answer, thinking_tokens, hidden_steps, thinking_end = interleaved_recurrent_answer(
                **common,
                hidden_steps_per_token=settings["hidden_steps_per_token"],
                max_thinking_tokens=settings["max_thinking_tokens"],
                use_thinking_scaffold=True,
                on_interleaved_token=on_interleaved_token,
            )
            if on_interleaved_complete is not None:
                on_interleaved_complete(thinking_tokens, hidden_steps, thinking_end)
            return answer
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
        transcript = self.main_screen.query_one("#chat-transcript", RichLog)
        if stopped:
            transcript.write(
                Text(
                    "Generation stopped; turn was not added to history.",
                    style=self._theme_color("warning"),
                )
            )
        else:
            self.messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )
            entry = Text("Assistant\n", style=f"bold {self._theme_color('accent')}")
            entry.append(answer)
            transcript.write(entry)
        self._set_ready(True)
        self._set_status("Ready")
        self.main_screen.query_one("#chat-input", Input).focus()

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
                on_hidden_step=self._hidden_trace,
                on_interleaved_token=self._interleaved_trace,
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
        entry = Text("You\n", style=f"bold {self._theme_color('primary')}")
        entry.append(prompt)
        self.query_one("#chat-transcript", RichLog).write(entry)
        self.stop_requested = False
        self._set_ready(False)
        self._set_status(f"Generating · {MODE_LABELS[mode]}")
        self.run_chat_turn(prompt, mode, settings)

    def _prepare_benchmark(self, cases: Sequence[dict]) -> None:
        self.benchmark_cases = {case["id"]: case for case in cases}
        self.output_buffers.clear()
        self.output_diagnostics.clear()
        self.rendered_output_keys.clear()
        self.running_case = None
        self.running_mode = None
        self.viewed_case = cases[0]["id"] if cases else None
        self.follow_live = True
        self.benchmark_started = time.perf_counter()
        self.benchmark_timer.resume()
        self.benchmark_finished = None
        self.benchmark_state = "Running"
        self.expanded_pane = None
        table = self.query_one("#case-table", DataTable)
        table.clear()
        for case in cases:
            cells = [case["id"], *(["·"] * len(MODES))]
            table.add_row(*cells, key=case["id"])
        progress = self.query_one("#benchmark-progress", ProgressBar)
        progress.update(total=len(cases) * len(MODES), progress=0)
        self.query_one("#benchmark-saved", Static).update(f"Saving to {self.results_file}")
        self._refresh_summary()
        self._refresh_inspector()
        self._refresh_benchmark_progress()
        self._apply_benchmark_layout()

    def _mark_running(self, case: dict, mode: str) -> None:
        self.running_case = case["id"]
        self.running_mode = mode
        self.main_screen.query_one("#case-table", DataTable).update_cell(
            case["id"], mode, Text("▶", style=self._theme_color("primary"))
        )
        self.output_buffers[(case["id"], mode)] = ""
        if self.follow_live:
            self._follow_running()
        else:
            self._refresh_inspector()
        self._refresh_benchmark_progress()

    def _record_result(self, result: dict, completed: int, total: int) -> None:
        marker, _, color = self._result_state(result)
        # Final output may contain text the generation callback did not emit.
        key = (result["case_id"], result["mode"])
        self.output_buffers[key] = result.get("raw_output", self.output_buffers.get(key, ""))
        self.main_screen.query_one("#case-table", DataTable).update_cell(
            result["case_id"],
            result["mode"],
            Text(marker, style=color),
        )
        self.main_screen.query_one("#benchmark-progress", ProgressBar).update(progress=completed)
        self.main_screen.query_one("#benchmark-saved", Static).update(f"Saved {completed}/{total} results · {self.results_file}")
        self._refresh_summary()
        self._refresh_inspector()
        self._refresh_benchmark_progress()

    def _refresh_summary(self) -> None:
        table = self.main_screen.query_one("#summary-table", DataTable)
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
            table.update_cell(mode, "latency", f"{latency / 1000:.1f}" if total else "—")

    def _finish_benchmark(self, stopped: bool) -> None:
        self.benchmark_state = "Stopped" if stopped else "Complete"
        self.benchmark_finished = time.perf_counter()
        self.benchmark_timer.pause()
        self._set_ready(True)
        self._set_status("Benchmark stopped" if stopped else "Benchmark complete")
        self.main_screen.query_one("#benchmark-saved", Static).update(
            f"Saved {len(self.results)} results · {self.results_file}"
        )
        self._refresh_benchmark_progress()
        self._refresh_inspector()

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
                latent_metrics = {
                    "attempted": 0,
                    "completed": 0,
                    "entropy_stopped": False,
                    "thinking_tokens": 0,
                    "thinking_end_detected": False,
                    "rms": None,
                }
                stream, stream_metrics = self._stream_benchmark(mode, case["id"])

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
                        f"{state}\nTemperature: {step.temperature:.2f} · Support: {step.effective_support:.1f}",
                    )

                def hidden_step(index, rms: float) -> None:
                    latent_metrics["attempted"] = index
                    latent_metrics["completed"] = index
                    self.call_from_thread(
                        self._update_output_title,
                        mode,
                        f"Latent {index}/{settings['soft_steps']}\nHidden-state RMS: {rms:.4f}",
                    )

                def interleaved_token(index: int, text: str, hidden_steps: int, rms: float) -> None:
                    latent_metrics["thinking_tokens"] = index
                    latent_metrics["completed"] = hidden_steps
                    latent_metrics["thinking_end_detected"] = False
                    latent_metrics["rms"] = rms
                    self.call_from_thread(
                        self._update_output_title,
                        mode,
                        f"Thinking {index}/{settings['max_thinking_tokens']} · Hidden: {hidden_steps}\nHidden-state RMS: {rms:.4f}",
                    )

                def interleaved_complete(thinking: int, hidden: int, ended: bool) -> None:
                    latent_metrics.update(
                        thinking_tokens=thinking,
                        completed=hidden,
                        thinking_end_detected=ended,
                    )
                    self.call_from_thread(
                        self._update_output_title,
                        mode,
                        f"Thinking: {thinking} · Hidden: {hidden}\nRMS: {latent_metrics['rms']:.4f} · {'natural end' if ended else 'limit/stop'}"
                        if latent_metrics["rms"] is not None
                        else f"Thinking: {thinking} · Hidden: {hidden} · {'natural end' if ended else 'limit/stop'}",
                    )

                try:
                    raw = self._generate_mode(
                        mode,
                        case["prompt"],
                        settings,
                        stream,
                        on_soft_step=soft_step if mode == "soft_recurrent" else None,
                        on_hidden_step=hidden_step if mode == "hidden_recurrent" else None,
                        on_hidden_complete=(
                            lambda completed: latent_metrics.__setitem__("completed", completed)
                        ) if mode == "hidden_recurrent" else None,
                        on_interleaved_token=interleaved_token if mode == "interleaved_recurrent" else None,
                        on_interleaved_complete=interleaved_complete
                        if mode == "interleaved_recurrent" else None,
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
                    if mode in ("soft_recurrent", "hidden_recurrent")
                    else latent_metrics["completed"] if mode == "interleaved_recurrent" else None,
                    "thinking_tokens_completed": latent_metrics.get("thinking_tokens")
                    if mode == "interleaved_recurrent" else None,
                    "thinking_end_detected": latent_metrics.get("thinking_end_detected")
                    if mode == "interleaved_recurrent" else None,
                    "latent_entropy_stopped": latent_metrics["entropy_stopped"]
                    if mode == "soft_recurrent"
                    else None,
                    "latency_ms": round(latency_ms, 3),
                }
                self.results.append(result)
                self._write_results()
                self.call_from_thread(self._record_result, result, len(self.results), total)
        self.call_from_thread(self._finish_benchmark, self.stop_requested)

    def action_start_benchmark(self) -> None:
        if self.busy or self.model is None:
            return
        try:
            settings = self._read_benchmark_settings()
            cases = load_benchmark_cases(self.benchmark_file)
            difficulty = str(self.query_one("#benchmark-difficulty", Select).value)
            if difficulty != "all":
                cases = select_benchmark_cases(cases, difficulties=[difficulty])
            if not cases:
                raise ValueError("No benchmark cases match this difficulty")
        except (ValueError, OSError) as error:
            self.notify(str(error), severity="error")
            return
        if self.settings_open:
            self._toggle_settings()
        if self.prompt_open:
            self._close_prompt()
        self._refresh_config_summary()
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
            if self.benchmark_state == "Running":
                self.benchmark_state = "Stopping"
                self._refresh_benchmark_progress()

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
        elif event.button.id in ("settings", "settings-done"):
            self._toggle_settings()
        elif event.button.id == "follow-live":
            self.follow_live = not self.follow_live
            if self.follow_live:
                self._follow_running()
            else:
                self._refresh_inspector()
        elif event.button.id == "show-prompt":
            self._show_prompt()
        elif event.button.id == "close-prompt":
            self._close_prompt()
        elif event.button.id in ("show-cases", "show-summary"):
            self.overview_page = event.button.id.removeprefix("show-")
            self.compact_page = "overview"
            self._apply_benchmark_layout()
        elif event.button.id == "show-output":
            self.compact_page = "output"
            self._apply_benchmark_layout()
        elif event.button.id in ("show-baseline", "show-comparison"):
            self.output_page = event.button.id.removeprefix("show-")
            if self.expanded_pane:
                self.expanded_pane = self.output_page
            self._apply_benchmark_layout()
        elif event.button.id in ("expand-baseline", "expand-comparison"):
            pane = event.button.id.removeprefix("expand-")
            self.expanded_pane = None if self.expanded_pane == pane else pane
            self._apply_benchmark_layout()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._start_chat()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "benchmark-softmax-mode":
            self._sync_softmax_controls()
        elif event.select.id == "comparison-mode" and event.value != Select.BLANK:
            mode = str(event.value)
            if mode != self.comparison_mode:
                self.comparison_mode = mode
                self.follow_live = False
                self._refresh_inspector()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "benchmark-entropy-enabled":
            self.query_one("#benchmark-entropy-stop", Input).disabled = not event.value

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "case-table":
            return
        self.viewed_case = str(event.row_key.value)
        self.follow_live = False
        self.compact_page = "output"
        self._refresh_inspector()
        self._apply_benchmark_layout()


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
