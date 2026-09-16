import unittest

from textual.widgets import DataTable, Input, Log, Select

from neuralese_tui import (
    NeuraleseTUI,
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
            self.assertEqual(app.query_one("#summary-table", DataTable).row_count, 4)

    async def test_benchmark_softmax_controls_and_stream_log(self) -> None:
        app = NeuraleseTUI(autoload=False)
        async with app.run_test(size=(160, 50)):
            adaptive = app._read_benchmark_settings()
            self.assertIsNone(adaptive["soft_temperature"])
            self.assertEqual(adaptive["target_support"], 8.0)
            self.assertEqual(adaptive["soft_top_k"], 64)
            self.assertEqual(adaptive["max_new_tokens"], 512)

            app.query_one("#benchmark-softmax-mode", Select).value = "fixed"
            app.query_one("#benchmark-temperature", Input).value = "0.8"
            app._sync_softmax_controls()
            fixed = app._read_benchmark_settings()
            self.assertEqual(fixed["soft_temperature"], 0.8)

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
            app._mark_running(case, "soft_recurrent")
            app.results.append(result)
            app._record_result(result, 1, 4)
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


if __name__ == "__main__":
    unittest.main()
