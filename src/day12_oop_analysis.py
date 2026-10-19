"""
FLOWCAST DAY 12
Object-Oriented Baseline Coverage Analysis

Purpose:
1. Load Day 11 baseline investigation results.
2. Compare prediction coverage between models.
3. Identify roads with low prediction coverage.
4. Generate CSV and Markdown reports.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "reports" / "day11"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "day12"


# ============================================================
# CLASS: FLOWCAST COVERAGE ANALYZER
# ============================================================

class FlowCastCoverageAnalyzer:
    """
    Analyze baseline prediction coverage using OOP.

    The class loads existing investigation results,
    compares forecasting baselines, and flags roads
    whose prediction coverage is below a threshold.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        coverage_threshold: float = 95.0,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            input_dir: Directory containing Day 11 reports.
            output_dir: Directory for Day 12 reports.
            coverage_threshold: Minimum desired coverage (%).
        """

        if not 0 <= coverage_threshold <= 100:
            raise ValueError(
                "Coverage threshold must be between 0 and 100."
            )

        # Instance attributes
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.coverage_threshold = coverage_threshold

        # Expected input files
        self.summary_path = (
            self.input_dir / "missing_prediction_summary.csv"
        )

        self.road_path = (
            self.input_dir / "missing_prediction_by_road.csv"
        )

    # --------------------------------------------------------
    # METHOD 1: LOAD INPUT DATA
    # --------------------------------------------------------

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load baseline summary and road-level investigation data.
        """

        if not self.summary_path.exists():
            raise FileNotFoundError(
                f"Summary file not found: {self.summary_path}"
            )

        if not self.road_path.exists():
            raise FileNotFoundError(
                f"Road report not found: {self.road_path}"
            )

        summary = pd.read_csv(self.summary_path)
        by_road = pd.read_csv(self.road_path)

        required_summary_columns = {
            "model",
            "validation_rows",
            "valid_predictions",
            "missing_predictions",
            "prediction_coverage_pct",
        }

        required_road_columns = {
            "model",
            "road_id",
            "missing_predictions",
            "coverage_pct",
        }

        self._validate_columns(
            summary,
            required_summary_columns,
            "Baseline summary",
        )

        self._validate_columns(
            by_road,
            required_road_columns,
            "Road-level report",
        )

        return summary, by_road

    # --------------------------------------------------------
    # METHOD 2: VALIDATE DATA COLUMNS
    # --------------------------------------------------------

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        dataset_name: str,
    ) -> None:
        """
        Ensure that the input DataFrame contains required columns.
        """

        missing_columns = required_columns - set(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

    # --------------------------------------------------------
    # METHOD 3: COMPARE BASELINE MODELS
    # --------------------------------------------------------

    def compare_baselines(
        self,
        summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare prediction coverage across baseline models.

        The model with the highest coverage becomes
        the reference model for the comparison.
        """

        required_columns = {
            "model",
            "validation_rows",
            "valid_predictions",
            "missing_predictions",
            "prediction_coverage_pct",
        }

        self._validate_columns(
            summary,
            required_columns,
            "Baseline summary",
        )

        if summary.empty:
            raise ValueError(
                "Cannot compare an empty baseline summary."
            )

        result = summary.copy()

        numeric_columns = [
            "validation_rows",
            "valid_predictions",
            "missing_predictions",
            "prediction_coverage_pct",
        ]

        for column in numeric_columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            )

        if result["model"].isna().any():
            raise ValueError(
                "Baseline model names cannot be missing."
            )

        if result["model"].duplicated().any():
            raise ValueError(
                "Baseline summary contains duplicate model names."
            )

        if (
            result["validation_rows"] <= 0
        ).any():
            raise ValueError(
                "Validation row counts must be positive."
            )

        if (
            result["prediction_coverage_pct"].isna().any()
            or (
                (result["prediction_coverage_pct"] < 0)
                | (result["prediction_coverage_pct"] > 100)
            ).any()
        ):
            raise ValueError(
                "Prediction coverage must be between 0 and 100."
            )

        # Highest-coverage baseline
        best_coverage = result[
            "prediction_coverage_pct"
        ].max()

        best_models = result.loc[
            result["prediction_coverage_pct"] == best_coverage,
            "model",
        ]

        # Add comparison columns
        result["best_coverage_pct"] = best_coverage

        result["coverage_gap_pp"] = (
            best_coverage
            - result["prediction_coverage_pct"]
        ).round(4)

        result["is_best_coverage"] = (
            result["prediction_coverage_pct"]
            == best_coverage
        )

        result["coverage_status"] = result[
            "prediction_coverage_pct"
        ].apply(
            lambda coverage: (
                "MEETS_THRESHOLD"
                if coverage >= self.coverage_threshold
                else "BELOW_THRESHOLD"
            )
        )

        result = result.sort_values(
            by="prediction_coverage_pct",
            ascending=False,
        ).reset_index(drop=True)

        print(
            "Highest-coverage baseline:",
            ", ".join(best_models.astype(str)),
        )

        return result

    # --------------------------------------------------------
    # METHOD 4: IDENTIFY LOW-COVERAGE ROADS
    # --------------------------------------------------------

    def flag_low_coverage_roads(
        self,
        by_road: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Identify road/model combinations below the coverage threshold.
        """

        required_columns = {
            "model",
            "road_id",
            "missing_predictions",
            "coverage_pct",
        }

        self._validate_columns(
            by_road,
            required_columns,
            "Road-level report",
        )

        result = by_road.copy()

        result["coverage_pct"] = pd.to_numeric(
            result["coverage_pct"],
            errors="raise",
        )

        if result["coverage_pct"].isna().any():
            raise ValueError(
                "Road coverage contains missing values."
            )

        if (
            (result["coverage_pct"] < 0)
            | (result["coverage_pct"] > 100)
        ).any():
            raise ValueError(
                "Road coverage must be between 0 and 100."
            )

        # Mark roads that do not meet the threshold.
        result["coverage_threshold_pct"] = (
            self.coverage_threshold
        )

        result["coverage_flag"] = result[
            "coverage_pct"
        ].apply(
            lambda coverage: (
                "LOW_COVERAGE"
                if coverage < self.coverage_threshold
                else "ACCEPTABLE_COVERAGE"
            )
        )

        # Sort lowest coverage first.
        result = result.sort_values(
            by=["coverage_pct", "missing_predictions"],
            ascending=[True, False],
        ).reset_index(drop=True)

        return result

    # --------------------------------------------------------
    # METHOD 5: BUILD MARKDOWN REPORT
    # --------------------------------------------------------

    @staticmethod
    def _markdown_table(
        dataframe: pd.DataFrame,
    ) -> str:
        """
        Convert a DataFrame to a Markdown table
        without requiring additional packages.
        """

        if dataframe.empty:
            return "_No records available._"

        columns = [
            str(column)
            for column in dataframe.columns
        ]

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(
                ["---"] * len(columns)
            ) + " |",
        ]

        for row in dataframe.itertuples(
            index=False,
            name=None,
        ):
            values = [
                "" if pd.isna(value) else str(value)
                for value in row
            ]

            lines.append(
                "| " + " | ".join(values) + " |"
            )

        return "\n".join(lines)

    def build_report(
        self,
        comparison: pd.DataFrame,
        road_flags: pd.DataFrame,
    ) -> str:
        """
        Create the Day 12 engineering report.
        """

        low_coverage = road_flags[
            road_flags["coverage_flag"] == "LOW_COVERAGE"
        ]

        best_models = comparison.loc[
            comparison["is_best_coverage"],
            "model",
        ].astype(str).tolist()

        report_sections = [
            "# FlowCast Day 12: OOP Baseline Coverage Analysis",
            "",
            "## 1. Objective",
            "",
            (
                "Apply object-oriented Python design to compare "
                "baseline prediction coverage and identify roads "
                "requiring further investigation."
            ),
            "",
            "## 2. Configuration",
            "",
            f"- Coverage threshold: {self.coverage_threshold:.2f}%",
            f"- Baselines evaluated: {len(comparison)}",
            f"- Road/model records examined: {len(road_flags)}",
            f"- Low-coverage road/model records: {len(low_coverage)}",
            "",
            "## 3. Baseline comparison",
            "",
            self._markdown_table(comparison),
            "",
            "## 4. Low-coverage road analysis",
            "",
            (
                "Road/model records are flagged when their "
                "coverage is strictly below the configured "
                "threshold."
            ),
            "",
            self._markdown_table(
                low_coverage.head(20)
            ),
            "",
            "## 5. Findings",
            "",
            (
                "Highest-coverage baseline(s): "
                + ", ".join(best_models)
            ),
            "",
            (
                f"{len(low_coverage)} road/model records "
                f"fall below the {self.coverage_threshold:.2f}% "
                "coverage threshold."
            ),
            "",
            "## 6. Interpretation",
            "",
            (
                "Prediction coverage measures how many validation "
                "rows received predictions. It does not measure "
                "forecast accuracy. A model with higher coverage "
                "may still have larger prediction errors."
            ),
            "",
            (
                "Low coverage identifies records for additional "
                "investigation. It does not, by itself, establish "
                "the cause of missing historical references."
            ),
            "",
            "## 7. Output files",
            "",
            "- `baseline_coverage_comparison.csv`",
            "- `road_coverage_flags.csv`",
            "- `day12_oop_analysis_report.md`",
            "",
        ]

        return "\n".join(report_sections)

    # --------------------------------------------------------
    # METHOD 6: SAVE OUTPUTS
    # --------------------------------------------------------

    def run(self) -> None:
        """
        Execute the complete Day 12 analysis.
        """

        print("=" * 65)
        print("FLOWCAST DAY 12")
        print("OBJECT-ORIENTED BASELINE COVERAGE ANALYSIS")
        print("=" * 65)

        print("\nLoading Day 11 reports...")

        summary, by_road = self.load_data()

        print("Baseline records:", len(summary))
        print("Road/model records:", len(by_road))

        print("\nComparing baseline models...")

        comparison = self.compare_baselines(summary)

        print("\nFlagging low-coverage roads...")

        road_flags = self.flag_low_coverage_roads(
            by_road
        )

        low_coverage = road_flags[
            road_flags["coverage_flag"] == "LOW_COVERAGE"
        ]

        print(
            "Low-coverage road/model records:",
            len(low_coverage),
        )

        # Create the output directory if it does not exist.
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save the comparison.
        comparison_path = (
            self.output_dir
            / "baseline_coverage_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        # Save road-level flags.
        road_flags_path = (
            self.output_dir
            / "road_coverage_flags.csv"
        )

        road_flags.to_csv(
            road_flags_path,
            index=False,
        )

        # Generate the Markdown report.
        report = self.build_report(
            comparison,
            road_flags,
        )

        report_path = (
            self.output_dir
            / "day12_oop_analysis_report.md"
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        print("\n" + "=" * 65)
        print("DAY 12 COMPLETED")
        print("=" * 65)

        print("\nBASELINE COMPARISON")
        print(
            comparison[
                [
                    "model",
                    "prediction_coverage_pct",
                    "coverage_gap_pp",
                    "coverage_status",
                    "is_best_coverage",
                ]
            ].to_string(
                index=False,
                col_space=12,
            )
        )

        print("\nREPORTS SAVED")
        print(comparison_path)
        print(road_flags_path)
        print(report_path)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

def main() -> None:
    """
    Create the analyzer object and run Day 12.
    """

    analyzer = FlowCastCoverageAnalyzer(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        coverage_threshold=95.0,
    )

    analyzer.run()


if __name__ == "__main__":
    main()