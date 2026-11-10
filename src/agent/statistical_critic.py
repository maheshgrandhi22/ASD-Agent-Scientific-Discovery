import math
import warnings

from scipy import stats


class StatisticalCritic:
    """Evaluate paired prediction errors and statistical assumptions."""

    SMALL_SAMPLE_THRESHOLD = 30
    SHAPIRO_MIN_SAMPLE_SIZE = 3
    SHAPIRO_MAX_SAMPLE_SIZE = 5000

    def _normality_diagnostic(self, improvements, alpha):
        sample_size = len(improvements)

        diagnostic = {
            "method": "shapiro_wilk",
            "status": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "alpha": float(alpha),
            "interpretation": (
                "At least three paired observations are required "
                "for the Shapiro-Wilk diagnostic."
            ),
        }

        if sample_size < self.SHAPIRO_MIN_SAMPLE_SIZE:
            return diagnostic

        if sample_size > self.SHAPIRO_MAX_SAMPLE_SIZE:
            diagnostic.update({
                "status": "skipped_sample_limit",
                "interpretation": (
                    "Shapiro-Wilk was skipped because the sample "
                    "exceeds 5000 observations."
                ),
            })
            return diagnostic

        if all(value == improvements[0] for value in improvements):
            diagnostic.update({
                "status": "constant_data",
                "interpretation": (
                    "All paired improvements are identical. "
                    "Normality cannot be meaningfully assessed."
                ),
            })
            return diagnostic

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                test = stats.shapiro(improvements)

            statistic = float(test.statistic)
            p_value = float(test.pvalue)

            if not math.isfinite(statistic) or not math.isfinite(p_value):
                diagnostic.update({
                    "status": "calculation_failed",
                    "interpretation": (
                        "The normality diagnostic produced "
                        "a non-finite result."
                    ),
                })
                return diagnostic

            diagnostic.update({
                "status": (
                    "evidence_against_normality"
                    if p_value < alpha
                    else "no_significant_evidence_against_normality"
                ),
                "statistic": statistic,
                "p_value": p_value,
                "interpretation": (
                    "The test provides evidence against the "
                    "normality assumption."
                    if p_value < alpha
                    else (
                        "The test did not detect significant evidence "
                        "against normality. This does not prove "
                        "that the data are normally distributed."
                    )
                ),
            })

        except (ValueError, FloatingPointError):
            diagnostic.update({
                "status": "calculation_failed",
                "interpretation": (
                    "The normality diagnostic could not be calculated."
                ),
            })

        return diagnostic

    def _wilcoxon_test(self, improvements, alpha):
        """Evaluate paired improvements with a Wilcoxon signed-rank test."""

        sample_size = len(improvements)
        zero_differences = sum(value == 0 for value in improvements)
        nonzero_pairs = sample_size - zero_differences

        result = {
            "method": "wilcoxon_signed_rank",
            "status": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "alpha": float(alpha),
            "zero_differences": zero_differences,
            "nonzero_pairs": nonzero_pairs,
            "interpretation": (
                "At least two nonzero paired improvements are required."
            ),
        }

        if sample_size == 0:
            return result

        if nonzero_pairs == 0:
            result.update({
                "status": "not_significant",
                "statistic": 0.0,
                "p_value": 1.0,
                "interpretation": (
                    "All paired improvements are zero. "
                    "There is no observed difference between the methods."
                ),
            })
            return result

        if nonzero_pairs < 2:
            return result

        try:
            # The normal approximation is used because tied absolute
            # differences can make the exact calculation inappropriate.
            test = stats.wilcoxon(
                improvements,
                zero_method="wilcox",
                correction=True,
                alternative="two-sided",
                method="approx",
            )

            statistic = float(test.statistic)
            p_value = float(test.pvalue)

            if not math.isfinite(statistic) or not math.isfinite(p_value):
                result.update({
                    "status": "calculation_failed",
                    "interpretation": (
                        "The Wilcoxon test produced a non-finite result."
                    ),
                })
                return result

            significant = p_value < alpha

            if significant and sum(improvements) > 0:
                interpretation = (
                    "The Wilcoxon test indicates significantly lower "
                    "absolute prediction errors for the candidate."
                )
            elif significant:
                interpretation = (
                    "The Wilcoxon test indicates significantly higher "
                    "absolute prediction errors for the candidate."
                )
            else:
                interpretation = (
                    "The Wilcoxon test did not detect a statistically "
                    "significant paired difference at the selected "
                    "alpha level."
                )

            result.update({
                "status": (
                    "significant" if significant else "not_significant"
                ),
                "statistic": statistic,
                "p_value": p_value,
                "interpretation": interpretation,
            })

        except (ValueError, FloatingPointError, OverflowError):
            result.update({
                "status": "calculation_failed",
                "interpretation": (
                    "The Wilcoxon signed-rank test could not be calculated."
                ),
            })

        return result

    def evaluate(self, dataset, alpha=0.05):
        if not isinstance(dataset, list):
            raise TypeError("dataset must be a list of dictionaries.")

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError("alpha must be a number between 0 and 1.")

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        baseline_errors = []
        candidate_errors = []
        improvements = []
        rejected_rows = 0

        for row in dataset:
            if not isinstance(row, dict):
                rejected_rows += 1
                continue

            values = (
                row.get("observed"),
                row.get("baseline_prediction"),
                row.get("candidate_prediction"),
            )

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                rejected_rows += 1
                continue

            observed, baseline, candidate = values

            baseline_error = abs(observed - baseline)
            candidate_error = abs(observed - candidate)

            baseline_errors.append(baseline_error)
            candidate_errors.append(candidate_error)
            improvements.append(baseline_error - candidate_error)

        sample_size = len(improvements)

        result = {
            "critic": "StatisticalCritic",
            "method": "paired_t_test_on_absolute_prediction_error",
            "status": "insufficient_data",
            "sample_size": sample_size,
            "rejected_rows": rejected_rows,
            "alpha": float(alpha),
            "confidence_level_percent": (1 - float(alpha)) * 100,
            "small_sample_warning": sample_size < self.SMALL_SAMPLE_THRESHOLD,
            "baseline_mae": None,
            "candidate_mae": None,
            "mean_improvement": None,
            "confidence_interval": None,
            "p_value": None,
            "statistic": None,
            "degrees_of_freedom": None,
            "normality_diagnostic": self._normality_diagnostic(
                improvements, alpha
            ),
            "wilcoxon_test": self._wilcoxon_test(
                improvements, alpha
            ),
            "interpretation": (
                "At least two valid paired observations are required."
            ),
            "assumptions": [
                "Each row contains a matched baseline and candidate prediction.",
                "Rows represent independent observations.",
                "The paired improvement values are sufficiently well "
                "approximated by a normal distribution for the paired t-test.",
            ],
            "limitations": [
                "A small sample may provide limited statistical power.",
                "Shapiro-Wilk does not prove that a distribution is normal.",
                "Normality testing does not establish independence.",
                "The Wilcoxon signed-rank test assumes independent pairs and "
                "approximately symmetric paired differences for its usual "
                "location interpretation.",
                "Statistical significance does not prove a scientific hypothesis.",
            ],
        }

        if sample_size == 0:
            return result

        result["baseline_mae"] = sum(baseline_errors) / sample_size
        result["candidate_mae"] = sum(candidate_errors) / sample_size
        result["mean_improvement"] = sum(improvements) / sample_size

        if sample_size < 2:
            return result

        mean_improvement = result["mean_improvement"]

        if all(value == 0 for value in improvements):
            result.update({
                "status": "not_significant",
                "confidence_interval": [0.0, 0.0],
                "p_value": 1.0,
                "statistic": 0.0,
                "degrees_of_freedom": sample_size - 1,
                "interpretation": (
                    "Baseline and candidate absolute errors are identical "
                    "for every valid observation."
                ),
            })
            return result

        standard_deviation = stats.tstd(improvements)

        if (
            not math.isfinite(float(standard_deviation))
            or standard_deviation == 0
        ):
            result["interpretation"] = (
                "The paired improvements have zero variance. "
                "A conventional t-test is not reported."
            )
            return result

        test = stats.ttest_1samp(
            improvements,
            popmean=0,
            alternative="two-sided",
        )

        standard_error = float(standard_deviation) / math.sqrt(sample_size)
        critical_value = float(
            stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
        )
        margin = critical_value * standard_error

        p_value = float(test.pvalue)
        statistic = float(test.statistic)
        lower = mean_improvement - margin
        upper = mean_improvement + margin

        if not all(
            math.isfinite(value)
            for value in (p_value, statistic, lower, upper)
        ):
            result["interpretation"] = (
                "The statistical calculation produced a non-finite result."
            )
            return result

        significant = p_value < alpha and (
            lower > 0 or upper < 0
        )

        result.update({
            "status": (
                "significant" if significant else "not_significant"
            ),
            "confidence_interval": [lower, upper],
            "p_value": p_value,
            "statistic": statistic,
            "degrees_of_freedom": sample_size - 1,
            "interpretation": (
                "The candidate has statistically significant lower "
                "absolute prediction error."
                if significant and mean_improvement > 0
                else (
                    "The candidate has statistically significant higher "
                    "absolute prediction error."
                    if significant
                    else (
                        "The data do not show a statistically significant "
                        "difference at the selected alpha level."
                    )
                )
            ),
        })

        return result
