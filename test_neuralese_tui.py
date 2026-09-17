import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from textual.widgets import DataTable, Input, Select, Static, Switch, TabbedContent
from textual.widgets.text_area import Selection

from neuralese_tui import (
    MODES,
    NeuraleseTUI,
    OutputText,
    parse_nonnegative_int,
    parse_positive_float,
    parse_positive_int,
    summarize_results,
)


class TuiHelperTests(unittest.TestCase):
    def test_numeric_settings_are_strict(self) -> None:
        self.assertEqual(parse_positive_int("8", "steps"), 8)
        self.assertEqual(parse_positive_float("4.5", "support"), 4.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_positive_int("0", "steps")
        with self.assertRaisesRegex(ValueError, "number"):
            parse_positive_float("many", "support")
        self.assertEqual(parse_nonnegative_int("0", "hidden steps"), 0)

    def test_result_summary_is_grouped_by_mode(self) -> None:
        results = [
            {"mode": "baseline", "correct": True, "latency_ms": 1000},
            {"mode": "baseline", "correct": False, "latency_ms": 3000},
            {"mode": "soft_recurrent", "correct": True, "latency_ms": 2000},
        ]
        rows = {row[0]: row for row in summarize_results(results)}
        self.assertEqual(rows["baseline"][1:4], (1, 2, 0.5))
        self.assertEqual(rows["baseline"][4], 2000)
        self.assertEqual(rows["soft_recurrent"][1:4], (1, 1, 1.0))
        self.assertEqual(rows["soft_recurrent"][5], 0.0)


class TuiStructureTests(unittest.IsolatedAsyncioTestCase):
    async def test_headless_layout_without_loading_model(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            self.assertEqual(len(app.query(DataTable)), 2)
            self.assertEqual(app.query_one("#chat-mode", Select).value, "soft_recurrent")
            self.assertEqual(app.query_one("#soft-steps", Input).value, "8")
            app.query_one("#interleaved-hidden-steps", Input).value = "0"
            self.assertEqual(app._read_chat_settings()["hidden_steps_per_token"], 0)
            app.query_one("#interleaved-hidden-steps", Input).value = "-1"
            with self.assertRaisesRegex(ValueError, "non-negative"):
                app._read_chat_settings()
            app.query_one("#interleaved-hidden-steps", Input).value = "0"
            app.query_one("#interleaved-thinking-tokens", Input).value = "0"
            with self.assertRaisesRegex(ValueError, "positive"):
                app._read_chat_settings()
            app.query_one("#interleaved-thinking-tokens", Input).value = "64"
            self.assertEqual(len(MODES), 6)
            self.assertEqual(app.query_one("#summary-table", DataTable).row_count, 6)
            self.assertEqual(len(app.query_one("#case-table", DataTable).columns), 7)
            self.assertEqual(len(app.query(".output-panel")), 2)

            # The width must be measured from the rendered content area, not
            # just the CSS width: Input borders and padding consume columns.
            # In particular, these values used to be reduced to two visible
            # digits by flex shrinking in the busy benchmark toolbar.
            chat_inputs = (
                "#soft-steps",
                "#target-support",
                "#max-tokens",
                "#interleaved-hidden-steps",
                "#interleaved-thinking-tokens",
            )
            for selector in chat_inputs:
                input_widget = app.query_one(selector, Input)
                self.assertGreaterEqual(
                    input_widget.content_region.width,
                    len(input_widget.value),
                    selector,
                )

            app.query_one(TabbedContent).active = "benchmark"
            await pilot.pause()
            for panel in app.query(".output-panel"):
                self.assertGreater(panel.region.width, 0)
                self.assertGreater(panel.region.height, 0)
            await pilot.click("#settings")
            await pilot.pause()
            benchmark_inputs = (
                "#benchmark-soft-steps",
                "#benchmark-support",
                "#benchmark-top-k",
                "#benchmark-min-temperature",
                "#benchmark-max-temperature",
                "#benchmark-entropy-stop",
                "#benchmark-max-tokens",
                "#benchmark-interleaved-hidden-steps",
                "#benchmark-interleaved-thinking-tokens",
            )
            for selector in benchmark_inputs:
                input_widget = app.query_one(selector, Input)
                self.assertGreaterEqual(
                    input_widget.content_region.width,
                    len(input_widget.value),
                    selector,
                )

            # A practical longer value should still fit without relying on a
            # user having to scroll the field just to read what they entered.
            longer = app.query_one("#benchmark-entropy-stop", Input)
            longer.value = "0.12345678"
            self.assertGreaterEqual(longer.content_region.width, len(longer.value))
            app.query_one("#benchmark-softmax-mode", Select).value = "fixed"
            await pilot.pause()
            fixed = app.query_one("#benchmark-temperature", Input)
            self.assertGreaterEqual(fixed.content_region.width, len(fixed.value))

    async def test_benchmark_softmax_controls_and_stream_buffer(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            adaptive = app._read_benchmark_settings()
            self.assertIsNone(adaptive["soft_temperature"])
            self.assertEqual(adaptive["target_support"], 8.0)
            self.assertEqual(adaptive["soft_top_k"], 64)
            self.assertEqual(adaptive["max_new_tokens"], 512)
            app.query_one("#benchmark-interleaved-hidden-steps", Input).value = "0"
            self.assertEqual(app._read_benchmark_settings()["hidden_steps_per_token"], 0)
            app.query_one("#benchmark-interleaved-hidden-steps", Input).value = "-1"
            with self.assertRaisesRegex(ValueError, "non-negative"):
                app._read_benchmark_settings()
            app.query_one("#benchmark-interleaved-hidden-steps", Input).value = "0"
            app.query_one("#benchmark-interleaved-thinking-tokens", Input).value = "0"
            with self.assertRaisesRegex(ValueError, "positive"):
                app._read_benchmark_settings()
            app.query_one("#benchmark-interleaved-thinking-tokens", Input).value = "64"
            app.query_one("#benchmark-interleaved-hidden-steps", Input).value = "2"

            app.query_one("#benchmark-softmax-mode", Select).value = "fixed"
            app.query_one("#benchmark-temperature", Input).value = "0.8"
            app._sync_softmax_controls()
            fixed = app._read_benchmark_settings()
            self.assertEqual(fixed["soft_temperature"], 0.8)
            self.assertEqual(fixed["hidden_steps_per_token"], 2)
            self.assertEqual(fixed["max_thinking_tokens"], 64)
            # Hidden adaptive values do not prevent a fixed-temperature run.
            app.query_one("#benchmark-min-temperature", Input).value = ""
            self.assertEqual(app._read_benchmark_settings()["soft_temperature"], 0.8)

            app.query_one("#benchmark-entropy-enabled", Switch).value = False
            await pilot.pause()
            self.assertTrue(app.query_one("#benchmark-entropy-stop", Input).disabled)
            self.assertIsNone(app._read_benchmark_settings()["entropy_stop"])
            app._prepare_benchmark([{"id": "demo", "category": "logic"}])
            app._append_output("demo", "baseline", "hello", False)
            app._append_output("demo", "baseline", " world\nnext", False)
            self.assertEqual(
                app.query_one("#output-baseline", OutputText).text,
                "hello world\nnext",
            )
            app._append_output("demo", "baseline", "replacement", True)
            self.assertEqual(app.query_one("#output-baseline", OutputText).text, "replacement")

    async def test_benchmark_tables_update_by_stable_keys(self) -> None:
        case = {"id": "demo", "category": "logic_constraints", "difficulty": "challenge"}
        result = {
            "case_id": "demo",
            "mode": "soft_recurrent",
            "normalized_answer": "x",
            "correct": True,
            "latency_ms": 1200,
        }
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            app._prepare_benchmark([case])
            row = app.query_one("#case-table", DataTable).get_row("demo")
            self.assertEqual(len(row), 7)
            self.assertEqual(
                app.query_one("#case-table", DataTable).get_cell("demo", "interleaved_recurrent"),
                "·",
            )
            self.assertEqual(app.query_one("#benchmark-progress").total, len(MODES))
            app._mark_running(case, "soft_recurrent")
            app.results.append(result)
            app._record_result(result, 1, len(MODES))
            await pilot.pause()
            self.assertTrue(
                any(
                    str(cell) == "✓"
                    for cell in app.query_one("#case-table", DataTable).get_row("demo")
                )
            )
            self.assertEqual(
                app.query_one("#summary-table", DataTable).get_row("soft_recurrent")[2],
                "100.0%",
            )

    async def test_responsive_dashboard_and_settings_are_accessible(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(200, 56)) as pilot:
            app.query_one(TabbedContent).active = "benchmark"
            case = {"id": "demo", "category": "logic"}
            app._prepare_benchmark([case])
            app._mark_running(case, "soft_recurrent")
            paragraph = (
                "We can solve this by tracking the constraints one at a time. "
                "Each assignment eliminates the remaining alternatives, so we must check "
                "that the final state satisfies every condition in the prompt.\n\n"
            )
            app._append_output("demo", "soft_recurrent", paragraph * 6 + "ANSWER: 42", False)
            for width, height in ((200, 56), (160, 50), (120, 40), (80, 24)):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                for selector in ("#run-benchmark", "#stop", "#settings", "#benchmark-status", "#output-comparison"):
                    widget = app.query_one(selector)
                    self.assertGreater(widget.region.width, 0, (width, height, selector))
                    self.assertLessEqual(widget.region.right, width, selector)
                    self.assertLessEqual(widget.region.bottom, height - 1, selector)
                output = app.query_one("#output-comparison", OutputText)
                self.assertGreaterEqual(output.content_region.height, 5)
                self.assertLessEqual(max(map(len, output.wrapped_document.get_sections(0))), output.wrap_width)
                # Check the displayed frame too: rewrapping the document alone
                # can leave Textual's compositor showing cropped cached lines.
                expected_line = output.render_line(0).text.strip()
                screenshot = ET.fromstring(app.export_screenshot())
                visible_text = [node.text.replace("\u00a0", " ").strip() for node in screenshot.iter()
                                if node.tag.endswith("text") and node.text]
                self.assertIn(expected_line, visible_text)
                self.assertEqual(app.query_one("#pane-baseline").display, width >= 140)
            await pilot.click("#show-summary")
            await pilot.pause()
            self.assertTrue(app.query_one("#benchmark-overview").display)
            self.assertFalse(app.query_one("#output-grid").display)
            self.assertGreaterEqual(app.query_one("#summary-table").content_region.height, 7)
            await pilot.click("#show-output")
            await pilot.click("#expand-comparison")
            await pilot.pause()
            self.assertGreater(app.query_one("#output-comparison").content_region.height, 10)
            await pilot.click("#expand-comparison")
            await pilot.click("#settings")
            app.query_one("#benchmark-interleaved-thinking-tokens").focus()
            await pilot.pause()
            field = app.query_one("#benchmark-interleaved-thinking-tokens", Input)
            self.assertGreater(field.region.y, 4)
            self.assertLess(field.region.bottom, 24)
            self.assertGreaterEqual(field.content_region.width, len(field.value))
            field.value = "0"
            await pilot.click("#settings")
            await pilot.pause()
            self.assertTrue(app.settings_open)
            self.assertIn("positive", str(app.query_one("#settings-error", Static).render()))
            field.value = "64"
            await pilot.click("#settings")
            self.assertFalse(app.settings_open)

    async def test_case_inspection_is_isolated_from_live_stream(self) -> None:
        cases = [
            {"id": "old_case_with_a_very_long_id", "category": "logic", "prompt": "Choose a letter", "accepted_answers": ["a"]},
            {"id": "new", "category": "logic", "prompt": "Choose a number", "accepted_answers": ["2"]},
        ]
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            app.query_one(TabbedContent).active = "benchmark"
            app._prepare_benchmark(cases)
            app._mark_running(cases[0], "baseline")
            app._append_output(cases[0]["id"], "baseline", "Old baseline", False)
            app._mark_running(cases[1], "soft_recurrent")
            app._append_output("new", "soft_recurrent", "New soft", False)
            await pilot.pause()
            self.assertTrue(app.follow_live)
            self.assertEqual(app.viewed_case, "new")
            # A pending baseline must not show output from the previous case.
            self.assertEqual(app.query_one("#output-baseline", OutputText).text, "")
            table = app.query_one("#case-table", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            self.assertFalse(app.follow_live)
            self.assertEqual(app.viewed_case, cases[0]["id"])
            self.assertEqual(app.query_one("#output-baseline", OutputText).text, "Old baseline")
            self.assertEqual(app.query_one("#output-comparison", OutputText).text, "")
            app._append_output("new", "soft_recurrent", " more tokens", False)
            app._update_output_title("soft_recurrent", "Generating 20 tokens", "new")
            self.assertEqual(app.query_one("#output-comparison", OutputText).text, "")
            await pilot.click("#show-prompt")
            self.assertTrue(app.prompt_open)
            # Progress and streamed output keep updating while the prompt is open.
            app._refresh_benchmark_progress()
            app._append_output("new", "soft_recurrent", " while reading", False)
            await pilot.click("#close-prompt")
            await pilot.click("#follow-live")
            self.assertEqual(app.viewed_case, "new")
            self.assertEqual(app.query_one("#output-comparison", OutputText).text, "New soft more tokens while reading")
            app.query_one("#comparison-mode", Select).value = "ordinary_cot"
            await pilot.pause()
            self.assertFalse(app.follow_live)
            self.assertEqual(app.query_one("#output-comparison", OutputText).text, "")

    async def test_stream_preserves_reading_position_and_resumes_at_tail(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            app.query_one(TabbedContent).active = "benchmark"
            case = {"id": "demo", "category": "logic"}
            other = {"id": "other", "category": "logic"}
            app._prepare_benchmark([case, other])
            app._mark_running(case, "baseline")
            app._append_output("demo", "baseline", "\n".join(f"line {i}" for i in range(100)), False)
            await pilot.pause()
            output = app.query_one("#output-baseline", OutputText)
            self.assertAlmostEqual(output.scroll_y, output.max_scroll_y)
            output.scroll_to(y=10, animate=False)
            await pilot.pause()
            app._append_output("demo", "baseline", "\nnext line", False)
            await pilot.pause()
            self.assertEqual(output.scroll_y, 10)
            output.scroll_end(animate=False)
            await pilot.pause()
            app._append_output("demo", "baseline", "\nlast line", False)
            await pilot.pause()
            self.assertAlmostEqual(output.scroll_y, output.max_scroll_y)
            app._mark_running(other, "baseline")
            await pilot.pause()
            table = app.query_one("#case-table", DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(output.scroll_y, 0)

    async def test_token_bursts_do_not_rebuild_document_or_scroll_backwards(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)) as pilot:
            app.query_one(TabbedContent).active = "benchmark"
            case = {"id": "demo", "category": "logic"}
            app._prepare_benchmark([case])
            app._mark_running(case, "baseline")
            text = "\n".join(f"Line {i}: text to inspect while more output arrives." for i in range(100))
            app._append_output("demo", "baseline", text, False)
            await pilot.pause()
            output = app.query_one("#output-baseline", OutputText)
            document = output.document
            selection = Selection((10, 8), (10, 12))
            output.selection = selection
            output.scroll_to(y=10, animate=False, immediate=True)
            await pilot.pause()

            transitions = []
            watch_scroll = output.watch_scroll_y

            def record_scroll(old, new):
                transitions.append((old, new))
                watch_scroll(old, new)

            with patch.object(output, "watch_scroll_y", new=record_scroll), patch.object(
                output, "load_text", wraps=output.load_text
            ) as reload_text:
                for index in range(12):
                    delta = f"\nMore output {index} " + "wrapping words " * 10
                    text += delta
                    app._append_output("demo", "baseline", delta, False)
                    if index % 3 == 2:
                        await pilot.pause()
                await pilot.pause()
                self.assertEqual(output.scroll_y, 10)
                self.assertTrue(all(old == new == 10 for old, new in transitions), transitions)
                self.assertEqual(output.selection, selection)

                output.scroll_end(animate=False, immediate=True)
                await pilot.pause()
                transitions.clear()
                for index in range(12):
                    delta = f"\nTail output {index}"
                    text += delta
                    app._append_output("demo", "baseline", delta, False)
                    if index % 3 == 2:
                        await pilot.pause()
                await pilot.pause()
                self.assertTrue(transitions)
                self.assertTrue(all(new >= old for old, new in transitions), transitions)
                self.assertEqual(output.scroll_y, output.max_scroll_y)

                # Corrected decoded suffixes also update in place.
                text = text[:-2] + "twelve"
                app._append_output("demo", "baseline", text, True)
                await pilot.pause()
                reload_text.assert_not_called()
                self.assertIs(output.document, document)
                self.assertEqual(output.selection, selection)
                self.assertEqual(output.text, text)
                screenshot = ET.fromstring(app.export_screenshot())
                visible_text = " ".join(node.text.replace("\u00a0", " ") for node in screenshot.iter()
                                        if node.tag.endswith("text") and node.text)
                self.assertIn("Tail output twelve", visible_text)

    async def test_worker_streams_saves_and_distinguishes_error_and_stop(self) -> None:
        case = {"id": "demo", "category": "logic", "difficulty": "challenge", "prompt": "Pick a letter", "accepted_answers": ["a"]}
        with TemporaryDirectory(prefix="neuralese-tui-") as directory:
            app = NeuraleseTUI(autoload=False, results_file=Path(directory) / "results.jsonl")
            async with app.run_test(size=(160, 50)) as pilot:
                app.query_one(TabbedContent).active = "benchmark"
                app.model = object()
                app._prepare_benchmark([case])
                app._set_ready(False)
                settings = app._read_benchmark_settings()
                await pilot.press("ctrl+p")
                self.assertIsNot(app.screen, app.main_screen)

                def generate(mode, prompt, settings, on_text, **kwargs):
                    if mode == "hard_argmax":
                        raise RuntimeError("example failure")
                    on_text("ANSWER: ")
                    on_text("ANSWER: a")
                    if mode == "interleaved_recurrent":
                        app.call_from_thread(app.action_request_stop)
                    return "ANSWER: a"

                with patch.object(app, "_generate_mode", side_effect=generate):
                    app.run_benchmark_worker([case], settings)
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                await pilot.press("escape")
                saved = [json.loads(line) for line in app.results_file.read_text().splitlines()]
                self.assertEqual(len(saved), 6)
                self.assertEqual(saved[0]["raw_output"], "ANSWER: a")
                self.assertEqual(saved[0]["output_tokens"], 2)
                self.assertEqual(app.benchmark_state, "Stopped")
                self.assertFalse(app.busy)
                table = app.query_one("#case-table", DataTable)
                self.assertEqual(str(table.get_cell("demo", "baseline")), "✓")
                self.assertEqual(str(table.get_cell("demo", "hard_argmax")), "!")
                self.assertEqual(str(table.get_cell("demo", "interleaved_recurrent")), "■")
                self.assertEqual(app.query_one("#output-comparison", OutputText).text, "ANSWER: a")

    async def test_hidden_mode_routes_and_reports_steps_without_model(self) -> None:
        app = NeuraleseTUI(autoload=False)
        app.model = object()
        app.tokenizer = object()
        settings = {
            "soft_steps": 3,
            "target_support": 8.0,
            "max_new_tokens": 16,
            "soft_temperature": None,
            "soft_top_k": 64,
            "entropy_stop": 0.75,
            "min_soft_temperature": 0.1,
            "max_soft_temperature": 4.0,
        }

        completed_steps = []

        def fake_hidden(**kwargs):
            kwargs["on_hidden_step"](1, 0.125)
            return "answer", 1

        with patch("neuralese_tui.hidden_recurrent_answer", side_effect=fake_hidden) as hidden:
            answer = app._generate_mode(
                "hidden_recurrent", "prompt", settings, lambda _: None,
                on_hidden_step=lambda *_: None,
                on_hidden_complete=completed_steps.append,
            )
        self.assertEqual(answer, "answer")
        hidden.assert_called_once()
        self.assertEqual(hidden.call_args.kwargs["latent_steps"], 3)
        self.assertEqual(completed_steps, [1])

    def test_interleaved_mode_routes_callback_and_counts(self) -> None:
        app = NeuraleseTUI(autoload=False)
        app.model = object()
        app.tokenizer = object()
        settings = {
            "soft_steps": 3, "target_support": 8.0, "max_new_tokens": 16,
            "soft_temperature": None, "soft_top_k": 64, "entropy_stop": 0.75,
            "min_soft_temperature": 0.1, "max_soft_temperature": 4.0,
            "hidden_steps_per_token": 2, "max_thinking_tokens": 64,
        }
        seen = []
        completed = []

        def fake_interleaved(**kwargs):
            kwargs["on_interleaved_token"](3, "step", 6, 0.25)
            return "answer", 3, 6, True

        with patch("neuralese_tui.interleaved_recurrent_answer", side_effect=fake_interleaved) as routed:
            answer = app._generate_mode(
                "interleaved_recurrent", "prompt", settings, lambda _: None,
                on_interleaved_token=lambda *args: seen.append(args),
                on_interleaved_complete=lambda *args: completed.append(args),
            )
        self.assertEqual(answer, "answer")
        routed.assert_called_once()
        self.assertEqual(seen, [(3, "step", 6, 0.25)])
        self.assertEqual(completed, [(3, 6, True)])
        self.assertEqual(routed.call_args.kwargs["hidden_steps_per_token"], 2)


if __name__ == "__main__":
    unittest.main()
