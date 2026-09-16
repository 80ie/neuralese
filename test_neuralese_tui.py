import unittest
from unittest.mock import patch

from textual.widgets import DataTable, Input, Log, Select, TabbedContent

from neuralese_tui import (
    MODES,
    NeuraleseTUI,
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
            self.assertEqual(len(app.query_one("#case-table", DataTable).columns), 8)
            self.assertEqual(len(app.query(".output-panel")), 6)

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

            # Only the active tab gets a rendered region, so activate the
            # benchmark toolbar before checking its actual usable width.
            app.query_one(TabbedContent).active = "benchmark"
            await pilot.pause()
            # The six panes use a two-column, three-row grid.
            for panel in app.query(".output-panel"):
                self.assertGreater(panel.region.width, 0)
                self.assertGreater(panel.region.height, 0)
            benchmark_inputs = (
                "#benchmark-soft-steps",
                "#benchmark-support",
                "#benchmark-temperature",
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

    async def test_benchmark_softmax_controls_and_stream_log(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)):
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

            app._append_output("#output-baseline", "hello", False)
            app._append_output("#output-baseline", " world\nnext", False)
            self.assertEqual(
                app.query_one("#output-baseline", Log).lines,
                ["hello world", "next"],
            )

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
            self.assertEqual(len(row), 8)
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
                    str(cell).startswith("✓ x")
                    for cell in app.query_one("#case-table", DataTable).get_row("demo")
                )
            )
            self.assertEqual(
                app.query_one("#summary-table", DataTable).get_row("soft_recurrent")[2],
                "100.0%",
            )

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
