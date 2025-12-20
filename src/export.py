"""
Results Export Module

Provides utilities for exporting radiomics results to various formats
including CSV, Excel, JSON, and generating analysis reports.
"""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ResultsExporter:
    """
    Exporter for radiomics analysis results.

    Provides methods for exporting feature data to various formats
    and generating summary reports.
    """

    FEATURE_CATEGORIES = {
        "firstorder": "First Order Statistics",
        "shape": "Shape Features",
        "glcm": "Gray Level Co-occurrence Matrix",
        "glrlm": "Gray Level Run Length Matrix",
        "glszm": "Gray Level Size Zone Matrix",
        "gldm": "Gray Level Dependence Matrix",
        "ngtdm": "Neighbouring Gray Tone Difference Matrix",
    }

    @staticmethod
    def to_csv(
        features: pd.DataFrame,
        path: Union[str, Path],
        include_diagnostics: bool = False,
    ) -> None:
        """
        Export features to CSV file.

        Args:
            features: DataFrame with radiomics features
            path: Output file path
            include_diagnostics: Whether to include diagnostic columns
        """
        df = features.copy()

        if not include_diagnostics:
            df = df[[col for col in df.columns if not col.startswith("diagnostics")]]

        df.to_csv(path, index=False)

    @staticmethod
    def to_csv_bytes(
        features: pd.DataFrame,
        include_diagnostics: bool = False,
    ) -> bytes:
        """
        Export features to CSV bytes (for download).

        Args:
            features: DataFrame with radiomics features
            include_diagnostics: Whether to include diagnostic columns

        Returns:
            CSV data as bytes
        """
        df = features.copy()

        if not include_diagnostics:
            df = df[[col for col in df.columns if not col.startswith("diagnostics")]]

        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def to_excel(
        features: pd.DataFrame,
        path: Union[str, Path],
        include_summary: bool = True,
        group_by_category: bool = True,
    ) -> None:
        """
        Export features to Excel file with multiple sheets.

        Args:
            features: DataFrame with radiomics features
            path: Output file path
            include_summary: Whether to include summary sheet
            group_by_category: Whether to group features by category
        """
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Main data sheet
            features.to_excel(writer, sheet_name="All Features", index=False)

            # Summary sheet
            if include_summary:
                summary = ResultsExporter._create_summary(features)
                summary.to_excel(writer, sheet_name="Summary", index=False)

            # Category sheets
            if group_by_category:
                categorized = ResultsExporter._categorize_features(features)
                for category, df in categorized.items():
                    if not df.empty:
                        sheet_name = category[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def to_excel_bytes(
        features: pd.DataFrame,
        include_summary: bool = True,
    ) -> bytes:
        """
        Export features to Excel bytes (for download).

        Args:
            features: DataFrame with radiomics features
            include_summary: Whether to include summary sheet

        Returns:
            Excel data as bytes
        """
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            features.to_excel(writer, sheet_name="All Features", index=False)

            if include_summary:
                summary = ResultsExporter._create_summary(features)
                summary.to_excel(writer, sheet_name="Summary", index=False)

        return buffer.getvalue()

    @staticmethod
    def to_json(
        features: pd.DataFrame,
        path: Union[str, Path],
        include_metadata: bool = True,
    ) -> None:
        """
        Export features to JSON file.

        Args:
            features: DataFrame with radiomics features
            path: Output file path
            include_metadata: Whether to include metadata
        """
        data = {
            "features": features.to_dict(orient="records"),
        }

        if include_metadata:
            data["metadata"] = {
                "export_date": datetime.now().isoformat(),
                "n_samples": len(features),
                "n_features": len(features.columns),
                "feature_names": list(features.columns),
            }

            # Include diagnostics if available
            if hasattr(features, "attrs") and "diagnostics" in features.attrs:
                data["diagnostics"] = features.attrs["diagnostics"]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    @staticmethod
    def to_json_bytes(features: pd.DataFrame) -> bytes:
        """
        Export features to JSON bytes (for download).

        Args:
            features: DataFrame with radiomics features

        Returns:
            JSON data as bytes
        """
        data = {
            "features": features.to_dict(orient="records"),
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "n_samples": len(features),
                "n_features": len(features.columns),
            },
        }

        return json.dumps(data, indent=2, cls=NumpyEncoder).encode("utf-8")

    @staticmethod
    def generate_report(
        features: pd.DataFrame,
        include_statistics: bool = True,
        include_correlations: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.

        Args:
            features: DataFrame with radiomics features
            include_statistics: Include descriptive statistics
            include_correlations: Include correlation analysis

        Returns:
            Dictionary with report data
        """
        report = {
            "summary": {
                "n_samples": len(features),
                "n_features": len(features.columns),
                "extraction_complete": True,
            },
            "feature_categories": {},
        }

        # Categorize features
        for col in features.columns:
            category = ResultsExporter._get_feature_category(col)
            if category:
                if category not in report["feature_categories"]:
                    report["feature_categories"][category] = {
                        "name": ResultsExporter.FEATURE_CATEGORIES.get(category, category),
                        "count": 0,
                        "features": [],
                    }
                report["feature_categories"][category]["count"] += 1
                report["feature_categories"][category]["features"].append(col)

        # Descriptive statistics
        if include_statistics:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            feature_cols = [
                col for col in numeric_cols
                if not col.startswith(("sample_", "diagnostics"))
            ]

            if feature_cols:
                stats = features[feature_cols].describe().to_dict()
                report["statistics"] = stats

        # Correlation analysis
        if include_correlations and len(features) > 1:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            feature_cols = [
                col for col in numeric_cols
                if not col.startswith(("sample_", "diagnostics"))
            ][:50]  # Limit for performance

            if len(feature_cols) > 1:
                corr_matrix = features[feature_cols].corr()

                # Find highly correlated pairs
                high_corr = []
                for i, col1 in enumerate(feature_cols):
                    for j, col2 in enumerate(feature_cols):
                        if i < j:
                            corr_val = corr_matrix.loc[col1, col2]
                            if abs(corr_val) > 0.9:
                                high_corr.append({
                                    "feature1": col1,
                                    "feature2": col2,
                                    "correlation": round(corr_val, 3),
                                })

                report["high_correlations"] = high_corr

        return report

    @staticmethod
    def _create_summary(features: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics DataFrame."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [
            col for col in numeric_cols
            if not col.startswith(("sample_", "diagnostics"))
        ]

        if not feature_cols:
            return pd.DataFrame()

        summary_data = []
        for col in feature_cols:
            category = ResultsExporter._get_feature_category(col)
            summary_data.append({
                "Feature": col,
                "Category": ResultsExporter.FEATURE_CATEGORIES.get(category, "Other"),
                "Mean": features[col].mean(),
                "Std": features[col].std(),
                "Min": features[col].min(),
                "Max": features[col].max(),
                "Median": features[col].median(),
            })

        return pd.DataFrame(summary_data)

    @staticmethod
    def _categorize_features(features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Categorize features by type."""
        categories = {}

        for category in ResultsExporter.FEATURE_CATEGORIES.keys():
            cols = [col for col in features.columns if category in col.lower()]
            if cols:
                categories[ResultsExporter.FEATURE_CATEGORIES[category]] = features[cols]

        return categories

    @staticmethod
    def _get_feature_category(name: str) -> Optional[str]:
        """Extract feature category from name."""
        name_lower = name.lower()
        for cat in ResultsExporter.FEATURE_CATEGORIES.keys():
            if cat in name_lower:
                return cat
        return None

    @staticmethod
    def format_feature_table(
        features: pd.DataFrame,
        max_rows: int = 100,
        precision: int = 4,
    ) -> pd.DataFrame:
        """
        Format features DataFrame for display.

        Args:
            features: DataFrame with radiomics features
            max_rows: Maximum rows to display
            precision: Decimal precision

        Returns:
            Formatted DataFrame
        """
        df = features.copy()

        # Round numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(precision)

        # Limit rows
        if len(df) > max_rows:
            df = df.head(max_rows)

        return df

    @staticmethod
    def create_feature_dictionary(features: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create a feature dictionary for documentation.

        Args:
            features: DataFrame with radiomics features

        Returns:
            List of feature dictionaries
        """
        feature_list = []

        for col in features.columns:
            if col.startswith(("sample_", "diagnostics")):
                continue

            category = ResultsExporter._get_feature_category(col)

            feature_info = {
                "name": col,
                "category": ResultsExporter.FEATURE_CATEGORIES.get(category, "Other"),
                "category_code": category,
            }

            # Add statistics if available
            if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_info.update({
                    "mean": float(features[col].mean()),
                    "std": float(features[col].std()),
                    "min": float(features[col].min()),
                    "max": float(features[col].max()),
                })

            feature_list.append(feature_info)

        return feature_list
