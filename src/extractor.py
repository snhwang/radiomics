"""
Radiomics Feature Extraction Module

This module provides a comprehensive interface for extracting radiomics features
from medical images using PyRadiomics with support for various image formats.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for radiomics feature extraction."""

    # Image preprocessing
    normalize: bool = True
    normalize_scale: int = 100
    resample_pixel_spacing: Optional[list] = None
    interpolator: str = "sitkBSpline"

    # Feature classes to extract
    enable_first_order: bool = True
    enable_shape: bool = True
    enable_glcm: bool = True
    enable_glrlm: bool = True
    enable_glszm: bool = True
    enable_gldm: bool = True
    enable_ngtdm: bool = True

    # GLCM settings
    glcm_distances: list = field(default_factory=lambda: [1])

    # Binning
    bin_width: int = 25
    bin_count: Optional[int] = None

    # Wavelet and LoG filters
    enable_wavelet: bool = False
    enable_log: bool = False
    log_sigma: list = field(default_factory=lambda: [1.0, 2.0, 3.0])

    # 2D vs 3D
    force_2d: bool = False
    force_2d_dimension: int = 0

    # Label settings
    label: int = 1

    def to_params(self) -> dict:
        """Convert configuration to PyRadiomics parameter dictionary."""
        params = {
            "setting": {
                "normalize": self.normalize,
                "normalizeScale": self.normalize_scale,
                "binWidth": self.bin_width,
                "label": self.label,
                "interpolator": self.interpolator,
                "force2D": self.force_2d,
                "force2Ddimension": self.force_2d_dimension,
            },
            "featureClass": {},
            "imageType": {"Original": {}},
        }

        if self.resample_pixel_spacing:
            params["setting"]["resampledPixelSpacing"] = self.resample_pixel_spacing

        if self.bin_count:
            params["setting"]["binCount"] = self.bin_count

        # Feature classes
        if self.enable_first_order:
            params["featureClass"]["firstorder"] = None
        if self.enable_shape:
            params["featureClass"]["shape"] = None
        if self.enable_glcm:
            params["featureClass"]["glcm"] = {"distances": self.glcm_distances}
        if self.enable_glrlm:
            params["featureClass"]["glrlm"] = None
        if self.enable_glszm:
            params["featureClass"]["glszm"] = None
        if self.enable_gldm:
            params["featureClass"]["gldm"] = None
        if self.enable_ngtdm:
            params["featureClass"]["ngtdm"] = None

        # Image filters
        if self.enable_wavelet:
            params["imageType"]["Wavelet"] = {}
        if self.enable_log:
            params["imageType"]["LoG"] = {"sigma": self.log_sigma}

        return params


class RadiomicsExtractor:
    """
    Main radiomics feature extraction class.

    Provides a unified interface for extracting radiomics features from
    medical images with support for DICOM, NIfTI, and other formats.
    """

    SUPPORTED_FORMATS = [".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".dcm", ".png", ".jpg", ".jpeg"]

    FEATURE_CATEGORIES = {
        "firstorder": "First Order Statistics",
        "shape": "Shape Features",
        "glcm": "Gray Level Co-occurrence Matrix (GLCM)",
        "glrlm": "Gray Level Run Length Matrix (GLRLM)",
        "glszm": "Gray Level Size Zone Matrix (GLSZM)",
        "gldm": "Gray Level Dependence Matrix (GLDM)",
        "ngtdm": "Neighbouring Gray Tone Difference Matrix (NGTDM)",
    }

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the radiomics extractor.

        Args:
            config: Extraction configuration. Uses defaults if not provided.
        """
        self.config = config or ExtractionConfig()
        self._extractor = None
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize the PyRadiomics feature extractor."""
        params = self.config.to_params()
        self._extractor = featureextractor.RadiomicsFeatureExtractor()

        # Apply settings
        for key, value in params["setting"].items():
            self._extractor.settings[key] = value

        # Disable all features first
        self._extractor.disableAllFeatures()

        # Enable selected feature classes
        for feature_class, settings in params["featureClass"].items():
            if settings is None:
                self._extractor.enableFeatureClassByName(feature_class)
            else:
                self._extractor.enableFeatureClassByName(feature_class)
                for setting_key, setting_value in settings.items():
                    self._extractor.settings[setting_key] = setting_value

        # Configure image types
        self._extractor.disableAllImageTypes()
        for image_type, settings in params["imageType"].items():
            self._extractor.enableImageTypeByName(image_type, customArgs=settings)

    def update_config(self, config: ExtractionConfig) -> None:
        """Update extraction configuration and reinitialize extractor."""
        self.config = config
        self._initialize_extractor()

    def extract_features(
        self,
        image: Union[str, Path, sitk.Image, np.ndarray],
        mask: Union[str, Path, sitk.Image, np.ndarray],
        label: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Extract radiomics features from an image with a segmentation mask.

        Args:
            image: Path to image file, SimpleITK Image, or numpy array
            mask: Path to mask file, SimpleITK Image, or numpy array
            label: Label value in the mask (overrides config if provided)

        Returns:
            DataFrame with extracted features
        """
        # Convert inputs to SimpleITK images
        sitk_image = self._to_sitk_image(image)
        sitk_mask = self._to_sitk_image(mask, is_mask=True)

        # Validate mask
        sitk_mask = self._validate_mask(sitk_image, sitk_mask)

        # Set label if provided
        if label is not None:
            self._extractor.settings["label"] = label

        # Extract features
        try:
            result = self._extractor.execute(sitk_image, sitk_mask)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}")

        # Parse results
        features = self._parse_results(result)

        return features

    def extract_features_batch(
        self,
        image_mask_pairs: list[tuple],
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Extract features from multiple image-mask pairs.

        Args:
            image_mask_pairs: List of (image, mask, optional_label) tuples
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with features for all images
        """
        all_features = []
        total = len(image_mask_pairs)

        for idx, pair in enumerate(image_mask_pairs):
            image = pair[0]
            mask = pair[1]
            label = pair[2] if len(pair) > 2 else None

            try:
                features = self.extract_features(image, mask, label)
                features["sample_index"] = idx

                # Add filename if available
                if isinstance(image, (str, Path)):
                    features["image_file"] = Path(image).name
                if isinstance(mask, (str, Path)):
                    features["mask_file"] = Path(mask).name

                all_features.append(features)

            except Exception as e:
                logger.warning(f"Failed to extract features for sample {idx}: {e}")

            if progress_callback:
                progress_callback(idx + 1, total)

        if not all_features:
            return pd.DataFrame()

        return pd.concat(all_features, ignore_index=True)

    def _to_sitk_image(
        self,
        data: Union[str, Path, sitk.Image, np.ndarray],
        is_mask: bool = False,
    ) -> sitk.Image:
        """Convert various input types to SimpleITK Image."""
        if isinstance(data, sitk.Image):
            return data

        if isinstance(data, np.ndarray):
            image = sitk.GetImageFromArray(data)
            if is_mask:
                image = sitk.Cast(image, sitk.sitkUInt8)
            return image

        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Handle different file types
            if path.suffix.lower() == ".dcm":
                return self._read_dicom(path)
            else:
                image = sitk.ReadImage(str(path))
                if is_mask:
                    image = sitk.Cast(image, sitk.sitkUInt8)
                return image

        raise TypeError(f"Unsupported input type: {type(data)}")

    def _read_dicom(self, path: Path) -> sitk.Image:
        """Read DICOM file or directory."""
        if path.is_dir():
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(path))
            reader.SetFileNames(dicom_files)
            return reader.Execute()
        else:
            return sitk.ReadImage(str(path))

    def _validate_mask(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> sitk.Image:
        """Validate and resample mask to match image if necessary."""
        # Check if resampling is needed
        if (image.GetSize() != mask.GetSize() or
            image.GetSpacing() != mask.GetSpacing() or
            image.GetOrigin() != mask.GetOrigin()):

            logger.info("Resampling mask to match image geometry")

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask = resampler.Execute(mask)

        return mask

    def _parse_results(self, result: dict) -> pd.DataFrame:
        """Parse PyRadiomics results into a clean DataFrame."""
        features = {}
        diagnostics = {}

        for key, value in result.items():
            # Skip configuration keys
            if key.startswith("diagnostics"):
                diagnostics[key] = value
            elif not key.startswith("general"):
                # Convert numpy types to Python types
                if hasattr(value, "item"):
                    value = value.item()
                features[key] = value

        # Create DataFrame
        df = pd.DataFrame([features])

        # Store diagnostics as attributes
        df.attrs["diagnostics"] = diagnostics

        return df

    @staticmethod
    def get_available_features() -> dict:
        """Get all available feature classes and their features."""
        feature_classes = getFeatureClasses()
        available = {}

        for name, feature_class in feature_classes.items():
            instance = feature_class(None, None)
            features = instance.getFeatureNames()
            available[name] = list(features.keys())

        return available

    @staticmethod
    def get_feature_description(feature_class: str, feature_name: str) -> str:
        """Get description for a specific feature."""
        try:
            feature_classes = getFeatureClasses()
            if feature_class in feature_classes:
                instance = feature_classes[feature_class](None, None)
                descriptions = instance.getFeatureNames()
                return descriptions.get(feature_name, "No description available")
        except Exception:
            pass
        return "No description available"
