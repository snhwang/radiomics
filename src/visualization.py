"""
Radiomics Visualization Module

Provides comprehensive visualization tools for medical images, segmentation masks,
and radiomics feature analysis results.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


class RadiomicsVisualizer:
    """
    Visualization tools for radiomics analysis.

    Provides methods for visualizing medical images, segmentation overlays,
    and extracted radiomics features.
    """

    # Color schemes
    FEATURE_COLORS = {
        "firstorder": "#3498db",
        "shape": "#2ecc71",
        "glcm": "#e74c3c",
        "glrlm": "#9b59b6",
        "glszm": "#f39c12",
        "gldm": "#1abc9c",
        "ngtdm": "#e67e22",
    }

    CATEGORY_NAMES = {
        "firstorder": "First Order",
        "shape": "Shape",
        "glcm": "GLCM",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
        "gldm": "GLDM",
        "ngtdm": "NGTDM",
    }

    @staticmethod
    def plot_image_slice(
        image: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        title: str = "Medical Image",
        cmap: str = "gray",
        figsize: tuple = (8, 8),
    ) -> go.Figure:
        """
        Plot a 2D slice from a 3D medical image.

        Args:
            image: 3D numpy array
            slice_idx: Slice index (middle slice if None)
            axis: Axis along which to slice (0, 1, or 2)
            title: Plot title
            cmap: Colormap name
            figsize: Figure size

        Returns:
            Plotly figure object
        """
        if image.ndim == 2:
            slice_data = image
        elif image.ndim == 3:
            if slice_idx is None:
                slice_idx = image.shape[axis] // 2
            slice_data = np.take(image, slice_idx, axis=axis)
        else:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")

        fig = px.imshow(
            slice_data,
            color_continuous_scale="gray" if cmap == "gray" else cmap,
            title=title,
            aspect="equal",
        )

        fig.update_layout(
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            coloraxis_showscale=False,
        )

        return fig

    @staticmethod
    def plot_image_with_mask(
        image: np.ndarray,
        mask: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        title: str = "Image with Segmentation",
        mask_alpha: float = 0.4,
        mask_color: str = "red",
    ) -> go.Figure:
        """
        Plot image with segmentation mask overlay.

        Args:
            image: 3D numpy array of the image
            mask: 3D numpy array of the segmentation mask
            slice_idx: Slice index
            axis: Axis along which to slice
            title: Plot title
            mask_alpha: Transparency of the mask overlay
            mask_color: Color of the mask

        Returns:
            Plotly figure object
        """
        if image.ndim == 2:
            img_slice = image
            mask_slice = mask
        else:
            if slice_idx is None:
                slice_idx = image.shape[axis] // 2
            img_slice = np.take(image, slice_idx, axis=axis)
            mask_slice = np.take(mask, slice_idx, axis=axis)

        # Normalize image
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Create RGB image
        rgb_image = np.stack([img_norm] * 3, axis=-1)

        # Create mask overlay
        mask_binary = mask_slice > 0

        # Color mapping
        color_map = {
            "red": [1, 0, 0],
            "green": [0, 1, 0],
            "blue": [0, 0, 1],
            "yellow": [1, 1, 0],
            "cyan": [0, 1, 1],
            "magenta": [1, 0, 1],
        }
        mask_rgb = color_map.get(mask_color, [1, 0, 0])

        # Apply mask overlay
        for c in range(3):
            rgb_image[:, :, c] = np.where(
                mask_binary,
                rgb_image[:, :, c] * (1 - mask_alpha) + mask_rgb[c] * mask_alpha,
                rgb_image[:, :, c],
            )

        fig = px.imshow(
            (rgb_image * 255).astype(np.uint8),
            title=title,
            aspect="equal",
        )

        fig.update_layout(coloraxis_showscale=False)

        return fig

    @staticmethod
    def plot_multi_slice(
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_slices: int = 9,
        axis: int = 2,
        title: str = "Multi-Slice View",
    ) -> go.Figure:
        """
        Plot multiple slices from a 3D volume in a grid.

        Args:
            image: 3D numpy array
            mask: Optional segmentation mask
            n_slices: Number of slices to display
            axis: Axis along which to slice
            title: Plot title

        Returns:
            Plotly figure object
        """
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))

        # Calculate slice indices
        total_slices = image.shape[axis]
        slice_indices = np.linspace(0, total_slices - 1, n_slices, dtype=int)

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Slice {i}" for i in slice_indices],
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
        )

        for idx, slice_idx in enumerate(slice_indices):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            img_slice = np.take(image, slice_idx, axis=axis)

            # Normalize
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            fig.add_trace(
                go.Heatmap(
                    z=img_norm,
                    colorscale="gray",
                    showscale=False,
                ),
                row=row,
                col=col,
            )

            # Add mask contour if provided
            if mask is not None:
                mask_slice = np.take(mask, slice_idx, axis=axis)
                if mask_slice.any():
                    fig.add_trace(
                        go.Contour(
                            z=mask_slice.astype(float),
                            showscale=False,
                            contours=dict(
                                start=0.5,
                                end=0.5,
                                size=1,
                                coloring="lines",
                            ),
                            line=dict(color="red", width=2),
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(
            title=title,
            height=200 * n_rows,
            width=200 * n_cols,
        )

        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)

        return fig

    @classmethod
    def plot_feature_heatmap(
        cls,
        features: pd.DataFrame,
        top_n: int = 30,
        title: str = "Feature Correlation Heatmap",
    ) -> go.Figure:
        """
        Plot correlation heatmap of radiomics features.

        Args:
            features: DataFrame with radiomics features
            top_n: Number of top features to include
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Select numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out metadata columns
        feature_cols = [
            col for col in numeric_cols
            if not col.startswith(("sample_", "diagnostics"))
        ][:top_n]

        if len(feature_cols) < 2:
            return go.Figure().add_annotation(
                text="Not enough features for correlation analysis",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Calculate correlation
        corr_matrix = features[feature_cols].corr()

        # Clean up feature names for display
        display_names = [cls._clean_feature_name(col) for col in feature_cols]

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=display_names,
                y=display_names,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            width=800,
            height=800,
            xaxis=dict(tickangle=45),
        )

        return fig

    @classmethod
    def plot_feature_distribution(
        cls,
        features: pd.DataFrame,
        feature_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot distribution of a single feature.

        Args:
            features: DataFrame with radiomics features
            feature_name: Name of the feature to plot
            title: Optional title

        Returns:
            Plotly figure object
        """
        if feature_name not in features.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame")

        display_name = cls._clean_feature_name(feature_name)
        category = cls._get_feature_category(feature_name)
        color = cls.FEATURE_COLORS.get(category, "#3498db")

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=features[feature_name],
                name=display_name,
                marker_color=color,
                opacity=0.8,
            )
        )

        fig.update_layout(
            title=title or f"Distribution of {display_name}",
            xaxis_title=display_name,
            yaxis_title="Count",
            showlegend=False,
        )

        return fig

    @classmethod
    def plot_feature_categories(
        cls,
        features: pd.DataFrame,
        title: str = "Features by Category",
    ) -> go.Figure:
        """
        Plot bar chart showing feature counts by category.

        Args:
            features: DataFrame with radiomics features
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Count features by category
        category_counts = {}
        for col in features.columns:
            category = cls._get_feature_category(col)
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1

        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = [cls.FEATURE_COLORS.get(cat, "#3498db") for cat in categories]
        display_names = [cls.CATEGORY_NAMES.get(cat, cat) for cat in categories]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=display_names,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Feature Category",
            yaxis_title="Number of Features",
        )

        return fig

    @classmethod
    def plot_feature_radar(
        cls,
        features: pd.DataFrame,
        selected_features: Optional[list] = None,
        normalize: bool = True,
        title: str = "Feature Radar Chart",
    ) -> go.Figure:
        """
        Plot radar chart of selected features.

        Args:
            features: DataFrame with radiomics features
            selected_features: List of feature names to include
            normalize: Whether to normalize features to [0, 1]
            title: Plot title

        Returns:
            Plotly figure object
        """
        if selected_features is None:
            # Select top features by variance
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            feature_cols = [
                col for col in numeric_cols
                if not col.startswith(("sample_", "diagnostics"))
            ]
            variances = features[feature_cols].var()
            selected_features = variances.nlargest(8).index.tolist()

        values = features[selected_features].iloc[0].values

        if normalize:
            min_vals = features[selected_features].min().values
            max_vals = features[selected_features].max().values
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            values = (values - min_vals) / range_vals

        display_names = [cls._clean_feature_name(f) for f in selected_features]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=list(values) + [values[0]],
                theta=display_names + [display_names[0]],
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.3)",
                line=dict(color="#3498db", width=2),
                name="Features",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else None,
                ),
            ),
            showlegend=False,
            title=title,
        )

        return fig

    @classmethod
    def plot_feature_boxplots(
        cls,
        features: pd.DataFrame,
        group_by: Optional[str] = None,
        top_n: int = 10,
        title: str = "Feature Box Plots",
    ) -> go.Figure:
        """
        Plot box plots for multiple features.

        Args:
            features: DataFrame with radiomics features
            group_by: Column name to group by
            top_n: Number of features to include
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Select top features by variance
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [
            col for col in numeric_cols
            if not col.startswith(("sample_", "diagnostics"))
        ][:top_n]

        if not feature_cols:
            return go.Figure()

        fig = go.Figure()

        for col in feature_cols:
            display_name = cls._clean_feature_name(col)
            category = cls._get_feature_category(col)
            color = cls.FEATURE_COLORS.get(category, "#3498db")

            # Normalize values for comparison
            values = features[col].values
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)

            fig.add_trace(
                go.Box(
                    y=normalized,
                    name=display_name,
                    marker_color=color,
                    boxmean=True,
                )
            )

        fig.update_layout(
            title=title,
            yaxis_title="Normalized Value",
            showlegend=False,
            height=500,
        )

        return fig

    @staticmethod
    def _clean_feature_name(name: str) -> str:
        """Clean feature name for display."""
        # Remove common prefixes
        prefixes = ["original_", "wavelet-", "log-sigma-"]
        for prefix in prefixes:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]

        # Split by underscore and capitalize
        parts = name.split("_")
        if len(parts) > 1:
            # Keep category short, capitalize feature name
            category = parts[0][:5]
            feature = "_".join(parts[1:])
            return f"{category}:{feature[:15]}"

        return name[:20]

    @staticmethod
    def _get_feature_category(name: str) -> Optional[str]:
        """Extract feature category from name."""
        categories = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
        name_lower = name.lower()
        for cat in categories:
            if cat in name_lower:
                return cat
        return None

    @staticmethod
    def create_histogram_distribution(
        data: np.ndarray,
        title: str = "Image Intensity Distribution",
        bins: int = 50,
    ) -> go.Figure:
        """Create histogram of image intensity distribution."""
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=data.flatten(),
                    nbinsx=bins,
                    marker_color="#3498db",
                    opacity=0.8,
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Intensity Value",
            yaxis_title="Count",
        )

        return fig
