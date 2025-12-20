"""
Image Preprocessing Module

Provides preprocessing utilities for medical images including normalization,
resampling, filtering, and enhancement techniques.
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    # Intensity normalization
    normalize_intensity: bool = True
    normalize_method: str = "zscore"  # zscore, minmax, histogram
    clip_percentiles: Optional[Tuple[float, float]] = (1, 99)

    # Resampling
    resample: bool = False
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolation: str = "linear"  # linear, nearest, bspline

    # Filtering
    apply_smoothing: bool = False
    smoothing_sigma: float = 1.0

    # Bias field correction
    correct_bias: bool = False
    bias_shrink_factor: int = 4

    # Windowing (for CT)
    apply_windowing: bool = False
    window_center: float = 40
    window_width: float = 400


class ImagePreprocessor:
    """
    Medical image preprocessing utilities.

    Provides methods for normalizing, resampling, and enhancing
    medical images before radiomics feature extraction.
    """

    INTERPOLATION_MAP = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
    }

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def preprocess(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
    ) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """
        Apply full preprocessing pipeline to image.

        Args:
            image: Input SimpleITK image
            mask: Optional segmentation mask

        Returns:
            Tuple of preprocessed image and mask
        """
        # Apply preprocessing steps in order
        if self.config.correct_bias:
            image = self.correct_bias_field(image, mask)

        if self.config.resample:
            image, mask = self.resample_image(
                image,
                mask,
                self.config.target_spacing,
            )

        if self.config.apply_windowing:
            image = self.apply_window(
                image,
                self.config.window_center,
                self.config.window_width,
            )

        if self.config.apply_smoothing:
            image = self.smooth_image(image, self.config.smoothing_sigma)

        if self.config.normalize_intensity:
            image = self.normalize(
                image,
                mask,
                method=self.config.normalize_method,
                clip_percentiles=self.config.clip_percentiles,
            )

        return image, mask

    def normalize(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        method: str = "zscore",
        clip_percentiles: Optional[Tuple[float, float]] = None,
    ) -> sitk.Image:
        """
        Normalize image intensities.

        Args:
            image: Input image
            mask: Optional mask for calculating statistics
            method: Normalization method (zscore, minmax, histogram)
            clip_percentiles: Percentiles for clipping outliers

        Returns:
            Normalized image
        """
        arr = sitk.GetArrayFromImage(image)

        # Get values within mask if provided
        if mask is not None:
            mask_arr = sitk.GetArrayFromImage(mask)
            values = arr[mask_arr > 0]
        else:
            values = arr.flatten()

        # Clip outliers
        if clip_percentiles:
            low, high = np.percentile(values, clip_percentiles)
            arr = np.clip(arr, low, high)

        # Apply normalization
        if method == "zscore":
            mean = values.mean()
            std = values.std()
            if std > 0:
                arr = (arr - mean) / std
            else:
                arr = arr - mean

        elif method == "minmax":
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val)
            else:
                arr = arr - min_val

        elif method == "histogram":
            arr = self._histogram_equalization(arr)

        # Create output image
        result = sitk.GetImageFromArray(arr)
        result.CopyInformation(image)

        return result

    def resample_image(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """
        Resample image to target spacing.

        Args:
            image: Input image
            mask: Optional segmentation mask
            target_spacing: Target voxel spacing

        Returns:
            Tuple of resampled image and mask
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Calculate new size
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(len(original_size))
        ]

        # Resample image
        interpolator = self.INTERPOLATION_MAP.get(
            self.config.interpolation,
            sitk.sitkLinear,
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)

        resampled_image = resampler.Execute(image)

        # Resample mask with nearest neighbor
        resampled_mask = None
        if mask is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_mask = resampler.Execute(mask)

        return resampled_image, resampled_mask

    def smooth_image(
        self,
        image: sitk.Image,
        sigma: float = 1.0,
    ) -> sitk.Image:
        """
        Apply Gaussian smoothing to image.

        Args:
            image: Input image
            sigma: Smoothing sigma in mm

        Returns:
            Smoothed image
        """
        smoothed = sitk.SmoothingRecursiveGaussian(image, sigma)
        return smoothed

    def correct_bias_field(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
    ) -> sitk.Image:
        """
        Apply N4 bias field correction.

        Args:
            image: Input image
            mask: Optional mask

        Returns:
            Bias-corrected image
        """
        # Cast to float
        image_float = sitk.Cast(image, sitk.sitkFloat32)

        # Shrink for speed
        shrink_factor = self.config.bias_shrink_factor
        shrunk_image = sitk.Shrink(
            image_float,
            [shrink_factor] * image_float.GetDimension(),
        )

        if mask is not None:
            shrunk_mask = sitk.Shrink(
                mask,
                [shrink_factor] * mask.GetDimension(),
            )
        else:
            shrunk_mask = sitk.OtsuThreshold(shrunk_image, 0, 1, 200)

        # Apply N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])

        corrected = corrector.Execute(shrunk_image, shrunk_mask)

        # Get bias field
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_float)
        bias_field = sitk.Exp(log_bias_field)

        # Apply correction to original resolution
        corrected_image = image_float / bias_field

        return corrected_image

    def apply_window(
        self,
        image: sitk.Image,
        center: float,
        width: float,
    ) -> sitk.Image:
        """
        Apply intensity windowing (typically for CT images).

        Args:
            image: Input image
            center: Window center (HU for CT)
            width: Window width

        Returns:
            Windowed image
        """
        low = center - width / 2
        high = center + width / 2

        windowed = sitk.IntensityWindowing(
            image,
            windowMinimum=low,
            windowMaximum=high,
            outputMinimum=0.0,
            outputMaximum=1.0,
        )

        return windowed

    @staticmethod
    def _histogram_equalization(arr: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        # Flatten and sort
        flat = arr.flatten()
        sorted_indices = np.argsort(flat)

        # Create equalized values
        equalized = np.zeros_like(flat)
        equalized[sorted_indices] = np.linspace(0, 1, len(flat))

        return equalized.reshape(arr.shape)

    @staticmethod
    def crop_to_mask(
        image: sitk.Image,
        mask: sitk.Image,
        margin: int = 10,
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        Crop image and mask to bounding box of the mask.

        Args:
            image: Input image
            mask: Segmentation mask
            margin: Margin around the mask in pixels

        Returns:
            Tuple of cropped image and mask
        """
        # Get bounding box
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        bounding_box = label_shape_filter.GetBoundingBox(1)

        # Add margin
        size = list(image.GetSize())
        start = list(bounding_box[:3])
        end = list(bounding_box[3:6])

        for i in range(3):
            start[i] = max(0, start[i] - margin)
            end[i] = min(size[i], start[i] + end[i] + 2 * margin)

        # Crop
        region_extractor = sitk.RegionOfInterestImageFilter()
        region_extractor.SetSize([end[i] - start[i] for i in range(3)])
        region_extractor.SetIndex(start)

        cropped_image = region_extractor.Execute(image)
        cropped_mask = region_extractor.Execute(mask)

        return cropped_image, cropped_mask

    @staticmethod
    def get_image_statistics(image: sitk.Image, mask: Optional[sitk.Image] = None) -> dict:
        """
        Get basic statistics of the image.

        Args:
            image: Input image
            mask: Optional mask

        Returns:
            Dictionary with image statistics
        """
        arr = sitk.GetArrayFromImage(image)

        if mask is not None:
            mask_arr = sitk.GetArrayFromImage(mask)
            values = arr[mask_arr > 0]
        else:
            values = arr.flatten()

        stats = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "shape": list(arr.shape),
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
            "dtype": str(arr.dtype),
        }

        return stats


# Common CT window presets
CT_WINDOW_PRESETS = {
    "lung": {"center": -600, "width": 1500},
    "mediastinum": {"center": 40, "width": 400},
    "bone": {"center": 400, "width": 2000},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 60, "width": 150},
    "soft_tissue": {"center": 50, "width": 350},
    "stroke": {"center": 40, "width": 40},
    "abdomen": {"center": 40, "width": 400},
}
