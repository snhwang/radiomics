"""
Segmentation Handler Module

Provides utilities for creating, loading, editing, and validating
segmentation masks for radiomics analysis.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


class SegmentationHandler:
    """
    Handler for segmentation masks in radiomics analysis.

    Provides methods for creating, loading, validating, and manipulating
    segmentation masks for use with radiomics feature extraction.
    """

    @staticmethod
    def create_mask_from_threshold(
        image: sitk.Image,
        lower: float,
        upper: float,
        fill_holes: bool = True,
        largest_component: bool = True,
    ) -> sitk.Image:
        """
        Create binary mask using intensity thresholding.

        Args:
            image: Input image
            lower: Lower threshold value
            upper: Upper threshold value
            fill_holes: Whether to fill holes in the mask
            largest_component: Whether to keep only largest connected component

        Returns:
            Binary segmentation mask
        """
        # Apply threshold
        mask = sitk.BinaryThreshold(
            image,
            lowerThreshold=lower,
            upperThreshold=upper,
            insideValue=1,
            outsideValue=0,
        )

        if fill_holes:
            mask = sitk.BinaryFillhole(mask)

        if largest_component:
            mask = SegmentationHandler.keep_largest_component(mask)

        return mask

    @staticmethod
    def create_mask_from_region(
        image: sitk.Image,
        seed_point: Tuple[int, ...],
        lower_threshold: float,
        upper_threshold: float,
    ) -> sitk.Image:
        """
        Create mask using region growing from seed point.

        Args:
            image: Input image
            seed_point: Starting point for region growing
            lower_threshold: Lower threshold for region growing
            upper_threshold: Upper threshold for region growing

        Returns:
            Binary segmentation mask
        """
        # Ensure seed point is tuple of ints
        seed_point = tuple(int(x) for x in seed_point)

        # Connected threshold
        mask = sitk.ConnectedThreshold(
            image,
            seedList=[seed_point],
            lower=lower_threshold,
            upper=upper_threshold,
            replaceValue=1,
        )

        return sitk.Cast(mask, sitk.sitkUInt8)

    @staticmethod
    def create_sphere_mask(
        reference_image: sitk.Image,
        center: Tuple[float, ...],
        radius: float,
    ) -> sitk.Image:
        """
        Create a spherical mask.

        Args:
            reference_image: Reference image for geometry
            center: Center of sphere in physical coordinates
            radius: Radius of sphere in mm

        Returns:
            Binary spherical mask
        """
        arr = sitk.GetArrayFromImage(reference_image)
        mask_arr = np.zeros_like(arr, dtype=np.uint8)

        # Convert center to index coordinates
        center_idx = reference_image.TransformPhysicalPointToIndex(center)

        # Get spacing
        spacing = reference_image.GetSpacing()

        # Create coordinate grids
        z, y, x = np.ogrid[
            0 : arr.shape[0], 0 : arr.shape[1], 0 : arr.shape[2]
        ]

        # Calculate distance from center (accounting for spacing)
        dist = np.sqrt(
            ((x - center_idx[0]) * spacing[0]) ** 2
            + ((y - center_idx[1]) * spacing[1]) ** 2
            + ((z - center_idx[2]) * spacing[2]) ** 2
        )

        mask_arr[dist <= radius] = 1

        mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(reference_image)

        return mask

    @staticmethod
    def create_cuboid_mask(
        reference_image: sitk.Image,
        corner1: Tuple[int, ...],
        corner2: Tuple[int, ...],
    ) -> sitk.Image:
        """
        Create a cuboidal (box) mask.

        Args:
            reference_image: Reference image for geometry
            corner1: First corner of the box (index coordinates)
            corner2: Opposite corner of the box (index coordinates)

        Returns:
            Binary cuboid mask
        """
        arr = sitk.GetArrayFromImage(reference_image)
        mask_arr = np.zeros_like(arr, dtype=np.uint8)

        # Ensure corners are ordered correctly
        z1, y1, x1 = corner1
        z2, y2, x2 = corner2

        z_min, z_max = min(z1, z2), max(z1, z2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        x_min, x_max = min(x1, x2), max(x1, x2)

        mask_arr[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = 1

        mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(reference_image)

        return mask

    @staticmethod
    def keep_largest_component(mask: sitk.Image) -> sitk.Image:
        """
        Keep only the largest connected component.

        Args:
            mask: Binary mask

        Returns:
            Mask with only largest component
        """
        # Label connected components
        labeled = sitk.ConnectedComponent(mask)

        # Get label statistics
        label_shape = sitk.LabelShapeStatisticsImageFilter()
        label_shape.Execute(labeled)

        # Find largest label
        labels = label_shape.GetLabels()
        if not labels:
            return mask

        largest_label = max(labels, key=lambda l: label_shape.GetNumberOfPixels(l))

        # Keep only largest
        result = sitk.BinaryThreshold(
            labeled,
            lowerThreshold=largest_label,
            upperThreshold=largest_label,
            insideValue=1,
            outsideValue=0,
        )

        return sitk.Cast(result, sitk.sitkUInt8)

    @staticmethod
    def dilate_mask(mask: sitk.Image, radius: int = 1) -> sitk.Image:
        """
        Dilate the mask.

        Args:
            mask: Binary mask
            radius: Dilation radius in voxels

        Returns:
            Dilated mask
        """
        return sitk.BinaryDilate(mask, [radius] * mask.GetDimension())

    @staticmethod
    def erode_mask(mask: sitk.Image, radius: int = 1) -> sitk.Image:
        """
        Erode the mask.

        Args:
            mask: Binary mask
            radius: Erosion radius in voxels

        Returns:
            Eroded mask
        """
        return sitk.BinaryErode(mask, [radius] * mask.GetDimension())

    @staticmethod
    def smooth_mask(mask: sitk.Image, sigma: float = 1.0) -> sitk.Image:
        """
        Smooth mask boundaries using Gaussian smoothing and re-thresholding.

        Args:
            mask: Binary mask
            sigma: Smoothing sigma

        Returns:
            Smoothed mask
        """
        # Convert to float
        mask_float = sitk.Cast(mask, sitk.sitkFloat32)

        # Smooth
        smoothed = sitk.SmoothingRecursiveGaussian(mask_float, sigma)

        # Re-threshold
        result = sitk.BinaryThreshold(
            smoothed, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0
        )

        return sitk.Cast(result, sitk.sitkUInt8)

    @staticmethod
    def get_mask_statistics(mask: sitk.Image, image: Optional[sitk.Image] = None) -> dict:
        """
        Get statistics about a segmentation mask.

        Args:
            mask: Binary mask
            image: Optional image for intensity statistics

        Returns:
            Dictionary with mask statistics
        """
        label_shape = sitk.LabelShapeStatisticsImageFilter()
        label_shape.Execute(mask)

        labels = label_shape.GetLabels()
        if not labels or 1 not in labels:
            return {"error": "No segmentation found in mask"}

        # Get spacing for volume calculation
        spacing = mask.GetSpacing()
        voxel_volume = np.prod(spacing)

        stats = {
            "voxel_count": label_shape.GetNumberOfPixels(1),
            "volume_mm3": label_shape.GetPhysicalSize(1),
            "volume_ml": label_shape.GetPhysicalSize(1) / 1000,
            "centroid": label_shape.GetCentroid(1),
            "bounding_box": label_shape.GetBoundingBox(1),
            "elongation": label_shape.GetElongation(1),
            "flatness": label_shape.GetFlatness(1),
            "roundness": label_shape.GetRoundness(1),
            "perimeter": label_shape.GetPerimeter(1) if mask.GetDimension() == 2 else None,
        }

        # Add intensity statistics if image provided
        if image is not None:
            label_intensity = sitk.LabelIntensityStatisticsImageFilter()
            label_intensity.Execute(mask, image)

            stats.update(
                {
                    "intensity_mean": label_intensity.GetMean(1),
                    "intensity_std": label_intensity.GetStandardDeviation(1),
                    "intensity_min": label_intensity.GetMinimum(1),
                    "intensity_max": label_intensity.GetMaximum(1),
                    "intensity_median": label_intensity.GetMedian(1),
                }
            )

        return stats

    @staticmethod
    def validate_mask(
        mask: sitk.Image,
        image: sitk.Image,
        min_voxels: int = 10,
    ) -> Tuple[bool, str]:
        """
        Validate a segmentation mask for radiomics analysis.

        Args:
            mask: Segmentation mask
            image: Corresponding image
            min_voxels: Minimum number of voxels required

        Returns:
            Tuple of (is_valid, message)
        """
        # Check dimensions
        if mask.GetDimension() != image.GetDimension():
            return False, "Mask and image have different dimensions"

        # Check size
        if mask.GetSize() != image.GetSize():
            return False, "Mask and image have different sizes"

        # Check for any segmentation
        mask_arr = sitk.GetArrayFromImage(mask)
        unique_labels = np.unique(mask_arr)

        if len(unique_labels) == 1 and unique_labels[0] == 0:
            return False, "Mask is empty (no segmentation)"

        # Check minimum voxels
        voxel_count = np.sum(mask_arr > 0)
        if voxel_count < min_voxels:
            return False, f"Mask has too few voxels ({voxel_count} < {min_voxels})"

        return True, "Mask is valid"

    @staticmethod
    def combine_masks(
        masks: List[sitk.Image],
        method: str = "union",
    ) -> sitk.Image:
        """
        Combine multiple masks.

        Args:
            masks: List of binary masks
            method: Combination method (union, intersection)

        Returns:
            Combined mask
        """
        if not masks:
            raise ValueError("No masks provided")

        if len(masks) == 1:
            return masks[0]

        result = masks[0]
        for mask in masks[1:]:
            if method == "union":
                result = sitk.Or(result, mask)
            elif method == "intersection":
                result = sitk.And(result, mask)
            else:
                raise ValueError(f"Unknown method: {method}")

        return result

    @staticmethod
    def split_multi_label_mask(mask: sitk.Image) -> dict:
        """
        Split a multi-label mask into separate binary masks.

        Args:
            mask: Multi-label mask

        Returns:
            Dictionary mapping label values to binary masks
        """
        arr = sitk.GetArrayFromImage(mask)
        labels = np.unique(arr)
        labels = labels[labels != 0]  # Exclude background

        masks = {}
        for label in labels:
            binary_arr = (arr == label).astype(np.uint8)
            binary_mask = sitk.GetImageFromArray(binary_arr)
            binary_mask.CopyInformation(mask)
            masks[int(label)] = binary_mask

        return masks

    @staticmethod
    def create_mask_from_contour(
        reference_image: sitk.Image,
        contour_points: np.ndarray,
        slice_idx: int,
        axis: int = 2,
    ) -> sitk.Image:
        """
        Create a mask from contour points on a single slice.

        Args:
            reference_image: Reference image for geometry
            contour_points: Nx2 array of contour points (x, y)
            slice_idx: Index of the slice
            axis: Axis perpendicular to the slice

        Returns:
            Binary mask with the contour filled
        """
        from skimage import draw

        arr = sitk.GetArrayFromImage(reference_image)
        mask_arr = np.zeros_like(arr, dtype=np.uint8)

        # Get the 2D slice shape
        slice_shape = np.delete(arr.shape, axis)

        # Create 2D mask from contour
        rr, cc = draw.polygon(contour_points[:, 1], contour_points[:, 0], slice_shape)

        # Apply to the correct slice
        if axis == 0:
            mask_arr[slice_idx, rr, cc] = 1
        elif axis == 1:
            mask_arr[rr, slice_idx, cc] = 1
        else:
            mask_arr[rr, cc, slice_idx] = 1

        mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(reference_image)

        return mask

    @staticmethod
    def load_mask(
        path: Union[str, Path],
        reference_image: Optional[sitk.Image] = None,
    ) -> sitk.Image:
        """
        Load a segmentation mask from file.

        Args:
            path: Path to mask file
            reference_image: Optional reference image for resampling

        Returns:
            Loaded mask
        """
        mask = sitk.ReadImage(str(path))
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Resample to match reference if provided
        if reference_image is not None:
            if mask.GetSize() != reference_image.GetSize():
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(reference_image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                mask = resampler.Execute(mask)

        return mask

    @staticmethod
    def save_mask(mask: sitk.Image, path: Union[str, Path]) -> None:
        """
        Save a segmentation mask to file.

        Args:
            mask: Mask to save
            path: Output path
        """
        sitk.WriteImage(mask, str(path))
