"""Evaluate paired predictions on explicitly designated holdout observations."""

import math


class HoldoutEvaluator:
    """Calculate descriptive prediction metrics on holdout rows only."""

    REQUIRED_FIELDS = (
        "observed",
        "baseline_prediction",
        "candidate_prediction",
    )
    VALID_SPLITS = {"train", "holdout"}

    def evaluate(self, dataset):
        if not isinstance(dataset, list):
            raise TypeError("dataset must be a list of dictionaries.")

        if not dataset:
            return self._empty_report("not_provided")

        has_split_column = any(
            isinstance(row, dict) and "data_split" in row
            for row in dataset
        )

        if not has_split_column:
            return self._empty_report("not_provided")

        train_rows = []
        holdout_rows = []
        invalid_split_rows = 0
        invalid_measurement_rows = 0

        for row in dataset:
            if not isinstance(row, dict):
                invalid_split_rows += 1
                continue

            split = row.get("data_split")

            if not isinstance(split, str):
                invalid_split_rows += 1
                continue

            split = split.strip().lower()

            if split not in self.VALID_SPLITS:
                invalid_split_rows += 1
                continue

            values = [row.get(field) for field in self.REQUIRED_FIELDS]

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                invalid_measurement_rows += 1
                continue

            clean_row = {
                field: float(row[field])
                for field in self.REQUIRED_FIELDS
            }

            if split == "train":
                train_rows.append(clean_row)
            else:
                holdout_rows.append(clean_row)

        report = {
            "evaluator": "HoldoutEvaluator",
            "method": "explicit_train_holdout_split",
            "status": "insufficient_holdout_data",
            "train_sample_size": len(train_rows),
            "holdout_sample_size": len(holdout_rows),
            "invalid_split_rows": invalid_split_rows,
            "invalid_measurement_rows": invalid_measurement_rows,
            "holdout_metrics": None,
            "scientific_validation": "not_performed",
            "split_provenance": (
                "Rows were assigned train/holdout labels by the dataset "
                "provider. The evaluator does not verify model-training "
                "provenance or guarantee absence of data leakage."
            ),
            "limitations": [
                "Holdout metrics describe only the supplied holdout rows.",
                "The evaluator cannot establish whether predictions were "
                "generated without access to holdout observations.",
                "Small holdout samples may produce unstable performance estimates.",
                "Predictive performance does not prove a scientific hypothesis.",
            ],
        }

        if len(holdout_rows) < 2:
            report["reason"] = (
                "At least two valid holdout observations are required "
                "for this evaluation."
            )
            return report

        errors = {
            "baseline": [
                row["observed"] - row["baseline_prediction"]
                for row in holdout_rows
            ],
            "candidate": [
                row["observed"] - row["candidate_prediction"]
                for row in holdout_rows
            ],
        }

        mae = {
            name: sum(abs(error) for error in values) / len(values)
            for name, values in errors.items()
        }

        rmse = {
            name: math.sqrt(
                sum(error ** 2 for error in values) / len(values)
            )
            for name, values in errors.items()
        }

        improvement_percent = (
            100 * (mae["baseline"] - mae["candidate"])
            / mae["baseline"]
            if mae["baseline"] > 0
            else None
        )

        report.update({
            "status": (
                "completed_small_sample"
                if len(holdout_rows) < 10
                else "completed"
            ),
            "holdout_metrics": {
                "mae": mae,
                "rmse": rmse,
                "candidate_mae_improvement_percent": (
                    round(improvement_percent, 2)
                    if improvement_percent is not None
                    else None
                ),
            },
        })

        if len(holdout_rows) < 10:
            report["reason"] = (
                "Holdout metrics were calculated, but fewer than "
                "10 valid holdout observations are available. "
                "Interpret the results cautiously."
            )

        return report

    @staticmethod
    def _empty_report(status):
        return {
            "evaluator": "HoldoutEvaluator",
            "method": "explicit_train_holdout_split",
            "status": status,
            "train_sample_size": 0,
            "holdout_sample_size": 0,
            "invalid_split_rows": 0,
            "invalid_measurement_rows": 0,
            "holdout_metrics": None,
            "scientific_validation": "not_performed",
            "reason": (
                "No explicit data_split column was supplied. "
                "Holdout evaluation was not performed."
            ),
        }
