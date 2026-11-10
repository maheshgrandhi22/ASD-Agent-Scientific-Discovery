import unittest

from src.agent.statistical_critic import StatisticalCritic


class TestStatisticalCritic(unittest.TestCase):

    def setUp(self):
        self.critic = StatisticalCritic()

    def test_detects_significant_candidate_improvement(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 10},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 20},
            {"observed": 30, "baseline_prediction": 26,
             "candidate_prediction": 30},
            {"observed": 40, "baseline_prediction": 35,
             "candidate_prediction": 40},
            {"observed": 50, "baseline_prediction": 44,
             "candidate_prediction": 50},
        ]

        result = self.critic.evaluate(dataset)

        self.assertEqual(result["status"], "significant")
        self.assertEqual(result["sample_size"], 5)
        self.assertGreater(result["mean_improvement"], 0)
        self.assertLess(result["p_value"], 0.05)
        self.assertGreater(result["confidence_interval"][0], 0)

    def test_detects_significant_candidate_regression(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 10,
             "candidate_prediction": 12},
            {"observed": 20, "baseline_prediction": 20,
             "candidate_prediction": 23},
            {"observed": 30, "baseline_prediction": 30,
             "candidate_prediction": 34},
            {"observed": 40, "baseline_prediction": 40,
             "candidate_prediction": 45},
            {"observed": 50, "baseline_prediction": 50,
             "candidate_prediction": 56},
        ]

        result = self.critic.evaluate(dataset)

        self.assertEqual(result["status"], "significant")
        self.assertLess(result["mean_improvement"], 0)
        self.assertLess(result["confidence_interval"][1], 0)

    def test_reports_no_difference_when_errors_are_identical(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 8},
            {"observed": 20, "baseline_prediction": 18,
             "candidate_prediction": 18},
        ]

        result = self.critic.evaluate(dataset)

        self.assertEqual(result["status"], "not_significant")
        self.assertEqual(result["p_value"], 1.0)

    def test_reports_insufficient_data_for_one_pair(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9}
        ]

        result = self.critic.evaluate(dataset)

        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["sample_size"], 1)
        self.assertIsNone(result["p_value"])

    def test_rejects_invalid_rows(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9},
            {"observed": float("nan"), "baseline_prediction": 8,
             "candidate_prediction": 9},
            {"observed": True, "baseline_prediction": 8,
             "candidate_prediction": 9},
            "invalid row",
        ]

        result = self.critic.evaluate(dataset)

        self.assertEqual(result["sample_size"], 1)
        self.assertEqual(result["rejected_rows"], 3)
        self.assertEqual(result["status"], "insufficient_data")

    def test_empty_dataset_is_insufficient(self):
        result = self.critic.evaluate([])

        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["sample_size"], 0)

    def test_invalid_dataset_type_is_rejected(self):
        with self.assertRaises(TypeError):
            self.critic.evaluate(None)

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            self.critic.evaluate([], alpha=1.0)


    def test_confidence_level_matches_custom_alpha(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 10},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 20},
            {"observed": 30, "baseline_prediction": 26,
             "candidate_prediction": 30},
            {"observed": 40, "baseline_prediction": 35,
             "candidate_prediction": 40},
            {"observed": 50, "baseline_prediction": 44,
             "candidate_prediction": 50},
        ]

        result = self.critic.evaluate(dataset, alpha=0.10)

        self.assertEqual(result["confidence_level_percent"], 90.0)
        self.assertIsNotNone(result["confidence_interval"])


    def test_reports_small_sample_warning(self):
        result = self.critic.evaluate([
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 10},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 20},
            {"observed": 30, "baseline_prediction": 26,
             "candidate_prediction": 30},
        ])
        self.assertTrue(result["small_sample_warning"])

    def test_shapiro_diagnostic_runs_with_three_pairs(self):
        result = self.critic.evaluate([
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 10},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 20},
            {"observed": 30, "baseline_prediction": 26,
             "candidate_prediction": 30},
        ])
        diagnostic = result["normality_diagnostic"]
        self.assertEqual(diagnostic["method"], "shapiro_wilk")
        self.assertIsNotNone(diagnostic["p_value"])

    def test_shapiro_diagnostic_handles_insufficient_pairs(self):
        result = self.critic.evaluate([
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9},
            {"observed": 20, "baseline_prediction": 17,
             "candidate_prediction": 19},
        ])
        self.assertEqual(
            result["normality_diagnostic"]["status"],
            "insufficient_data",
        )

    def test_shapiro_diagnostic_handles_constant_improvements(self):
        result = self.critic.evaluate([
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 9},
            {"observed": 20, "baseline_prediction": 18,
             "candidate_prediction": 19},
            {"observed": 30, "baseline_prediction": 28,
             "candidate_prediction": 29},
        ])
        self.assertEqual(
            result["normality_diagnostic"]["status"],
            "constant_data",
        )

    def test_shapiro_diagnostic_skips_more_than_5000_pairs(self):
        dataset = [
            {
                "observed": float(i),
                "baseline_prediction": float(i + 2),
                "candidate_prediction": float(i + 1),
            }
            for i in range(5001)
        ]
        result = self.critic.evaluate(dataset)
        self.assertEqual(
            result["normality_diagnostic"]["status"],
            "skipped_sample_limit",
        )

    def test_reports_statistical_limitations(self):
        result = self.critic.evaluate([])
        self.assertTrue(result["limitations"])
        self.assertTrue(
            any("independence" in item.lower()
                for item in result["limitations"])
        )


    def test_wilcoxon_detects_candidate_improvement(self):
        improvements = [3, 4, 3, 4, 4, 5]
        dataset = [
            {
                "observed": 100 + i,
                "baseline_prediction": 100 + i + value,
                "candidate_prediction": 100 + i,
            }
            for i, value in enumerate(improvements)
        ]

        result = self.critic.evaluate(dataset)
        wilcoxon = result["wilcoxon_test"]

        self.assertEqual(wilcoxon["status"], "significant")
        self.assertLess(wilcoxon["p_value"], 0.05)
        self.assertEqual(wilcoxon["nonzero_pairs"], 6)

    def test_wilcoxon_reports_no_difference_for_zero_improvements(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 8,
             "candidate_prediction": 8},
            {"observed": 20, "baseline_prediction": 18,
             "candidate_prediction": 18},
            {"observed": 30, "baseline_prediction": 28,
             "candidate_prediction": 28},
        ]

        result = self.critic.evaluate(dataset)
        wilcoxon = result["wilcoxon_test"]

        self.assertEqual(wilcoxon["status"], "not_significant")
        self.assertEqual(wilcoxon["p_value"], 1.0)
        self.assertEqual(wilcoxon["zero_differences"], 3)

    def test_wilcoxon_handles_zero_differences(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 15,
             "candidate_prediction": 12},
            {"observed": 20, "baseline_prediction": 20,
             "candidate_prediction": 20},
            {"observed": 30, "baseline_prediction": 34,
             "candidate_prediction": 31},
            {"observed": 40, "baseline_prediction": 48,
             "candidate_prediction": 44},
        ]

        result = self.critic.evaluate(dataset)
        wilcoxon = result["wilcoxon_test"]

        self.assertEqual(wilcoxon["zero_differences"], 1)
        self.assertEqual(wilcoxon["nonzero_pairs"], 3)
        self.assertIsNotNone(wilcoxon["p_value"])

    def test_wilcoxon_requires_two_nonzero_pairs(self):
        dataset = [
            {"observed": 10, "baseline_prediction": 15,
             "candidate_prediction": 12},
            {"observed": 20, "baseline_prediction": 20,
             "candidate_prediction": 20},
        ]

        result = self.critic.evaluate(dataset)
        wilcoxon = result["wilcoxon_test"]

        self.assertEqual(wilcoxon["status"], "insufficient_data")
        self.assertIsNone(wilcoxon["p_value"])

if __name__ == "__main__":
    unittest.main()

