import unittest

from src.agent.holdout_evaluator import HoldoutEvaluator


class TestHoldoutEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = HoldoutEvaluator()

    def test_missing_split_column_is_not_provided(self):
        dataset = [{
            "observed": 10,
            "baseline_prediction": 8,
            "candidate_prediction": 9,
        }]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["status"], "not_provided")
        self.assertIsNone(result["holdout_metrics"])

    def test_metrics_use_holdout_rows_only(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 0,
                "candidate_prediction": 10,
                "data_split": "train",
            },
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "holdout",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["status"], "completed_small_sample")
        self.assertEqual(result["train_sample_size"], 1)
        self.assertEqual(result["holdout_sample_size"], 2)
        self.assertEqual(
            result["holdout_metrics"]["mae"]["baseline"], 2.5
        )
        self.assertEqual(
            result["holdout_metrics"]["mae"]["candidate"], 1.0
        )
        self.assertEqual(
            result["holdout_metrics"]["rmse"]["baseline"],
            (6.5) ** 0.5,
        )

    def test_split_labels_are_case_insensitive(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "TRAIN",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "Holdout",
            },
            {
                "observed": 30,
                "baseline_prediction": 27,
                "candidate_prediction": 29,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["train_sample_size"], 1)
        self.assertEqual(result["holdout_sample_size"], 2)
        self.assertEqual(result["status"], "completed_small_sample")

    def test_invalid_split_labels_are_rejected(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "validation",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
            {
                "observed": 30,
                "baseline_prediction": 27,
                "candidate_prediction": 29,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["invalid_split_rows"], 1)
        self.assertEqual(result["holdout_sample_size"], 2)
        self.assertEqual(result["status"], "completed_small_sample")

    def test_missing_split_on_some_rows_is_rejected(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "train",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
            },
            {
                "observed": 30,
                "baseline_prediction": 27,
                "candidate_prediction": 29,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["invalid_split_rows"], 1)
        self.assertEqual(result["holdout_sample_size"], 1)
        self.assertEqual(result["status"], "insufficient_holdout_data")

    def test_invalid_numeric_measurements_are_rejected(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "holdout",
            },
            {
                "observed": 20,
                "baseline_prediction": float("nan"),
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
            {
                "observed": 30,
                "baseline_prediction": 27,
                "candidate_prediction": 29,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["invalid_measurement_rows"], 1)
        self.assertEqual(result["holdout_sample_size"], 2)
        self.assertEqual(result["status"], "completed_small_sample")

    def test_boolean_measurements_are_rejected(self):
        dataset = [
            {
                "observed": True,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "holdout",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "holdout",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["invalid_measurement_rows"], 1)
        self.assertEqual(result["holdout_sample_size"], 1)
        self.assertEqual(result["status"], "insufficient_holdout_data")

    def test_one_holdout_observation_is_insufficient(self):
        dataset = [{
            "observed": 10,
            "baseline_prediction": 8,
            "candidate_prediction": 9,
            "data_split": "holdout",
        }]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(
            result["status"], "insufficient_holdout_data"
        )
        self.assertIsNone(result["holdout_metrics"])

    def test_non_list_dataset_is_rejected(self):
        with self.assertRaises(TypeError):
            self.evaluator.evaluate({})

    def test_empty_dataset_is_not_provided(self):
        result = self.evaluator.evaluate([])

        self.assertEqual(result["status"], "not_provided")

    def test_small_holdout_sample_has_warning_status(self):
        dataset = [
            {
                "observed": 10 + index,
                "baseline_prediction": 8 + index,
                "candidate_prediction": 9 + index,
                "data_split": "holdout",
            }
            for index in range(2)
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(result["status"], "completed_small_sample")
        self.assertIn("reason", result)

    def test_no_holdout_rows_is_insufficient(self):
        dataset = [
            {
                "observed": 10,
                "baseline_prediction": 8,
                "candidate_prediction": 9,
                "data_split": "train",
            },
            {
                "observed": 20,
                "baseline_prediction": 17,
                "candidate_prediction": 19,
                "data_split": "train",
            },
        ]

        result = self.evaluator.evaluate(dataset)

        self.assertEqual(
            result["status"], "insufficient_holdout_data"
        )
        self.assertEqual(result["train_sample_size"], 2)
        self.assertEqual(result["holdout_sample_size"], 0)


if __name__ == "__main__":
    unittest.main()
