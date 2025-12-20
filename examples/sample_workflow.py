"""
Sample Radiomics Workflow

This script demonstrates how to use the Radiomics Platform programmatically
for extracting and analyzing radiomics features from medical images.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import RadiomicsExtractor, ExtractionConfig
from src.preprocessing import ImagePreprocessor, PreprocessingConfig
from src.segmentation import SegmentationHandler
from src.visualization import RadiomicsVisualizer
from src.export import ResultsExporter


def create_sample_data():
    """Create sample image and mask for demonstration."""
    # Create a simple 3D phantom image
    size = (64, 64, 32)

    # Create image with spherical structure
    image_arr = np.zeros(size, dtype=np.float32)
    center = np.array([32, 32, 16])

    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < 15:
                    image_arr[z, y, x] = 200 + np.random.normal(0, 20)
                else:
                    image_arr[z, y, x] = 50 + np.random.normal(0, 10)

    # Create mask
    mask_arr = np.zeros(size, dtype=np.uint8)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < 12:
                    mask_arr[z, y, x] = 1

    # Convert to SimpleITK images
    image = sitk.GetImageFromArray(image_arr)
    image.SetSpacing([1.0, 1.0, 2.0])

    mask = sitk.GetImageFromArray(mask_arr)
    mask.SetSpacing([1.0, 1.0, 2.0])

    return image, mask


def example_basic_extraction():
    """Basic feature extraction example."""
    print("=" * 60)
    print("Example 1: Basic Feature Extraction")
    print("=" * 60)

    # Create sample data
    image, mask = create_sample_data()
    print(f"Image size: {image.GetSize()}")
    print(f"Image spacing: {image.GetSpacing()}")

    # Create extractor with default settings
    extractor = RadiomicsExtractor()

    # Extract features
    print("\nExtracting features...")
    features = extractor.extract_features(image, mask)

    print(f"\nExtracted {len(features.columns)} features")
    print("\nFirst 10 features:")
    for col in list(features.columns)[:10]:
        print(f"  {col}: {features[col].values[0]:.4f}")

    return features


def example_custom_configuration():
    """Feature extraction with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Create sample data
    image, mask = create_sample_data()

    # Create custom configuration
    config = ExtractionConfig(
        normalize=True,
        bin_width=10,  # Finer binning
        enable_first_order=True,
        enable_shape=True,
        enable_glcm=True,
        enable_glrlm=False,  # Disable some features
        enable_glszm=False,
        enable_gldm=False,
        enable_ngtdm=False,
        glcm_distances=[1, 2],  # Multiple distances
    )

    # Create extractor
    extractor = RadiomicsExtractor(config)

    # Extract features
    print("Extracting with custom configuration...")
    features = extractor.extract_features(image, mask)

    print(f"\nExtracted {len(features.columns)} features")

    # Show feature categories
    categories = {}
    for col in features.columns:
        for cat in ["firstorder", "shape", "glcm"]:
            if cat in col.lower():
                categories[cat] = categories.get(cat, 0) + 1

    print("\nFeatures by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")

    return features


def example_preprocessing():
    """Image preprocessing before extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Image Preprocessing")
    print("=" * 60)

    # Create sample data
    image, mask = create_sample_data()

    # Configure preprocessing
    preprocess_config = PreprocessingConfig(
        normalize_intensity=True,
        normalize_method="zscore",
        apply_smoothing=True,
        smoothing_sigma=0.5,
        resample=True,
        target_spacing=(1.0, 1.0, 1.0),
    )

    # Apply preprocessing
    preprocessor = ImagePreprocessor(preprocess_config)
    processed_image, processed_mask = preprocessor.preprocess(image, mask)

    print("Original image:")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Size: {image.GetSize()}")

    print("\nProcessed image:")
    print(f"  Spacing: {processed_image.GetSpacing()}")
    print(f"  Size: {processed_image.GetSize()}")

    # Get statistics
    stats = ImagePreprocessor.get_image_statistics(processed_image, processed_mask)
    print("\nImage statistics (within mask):")
    for key in ["mean", "std", "min", "max"]:
        print(f"  {key}: {stats[key]:.4f}")

    return processed_image, processed_mask


def example_segmentation():
    """Segmentation utilities example."""
    print("\n" + "=" * 60)
    print("Example 4: Segmentation Utilities")
    print("=" * 60)

    # Create sample data
    image, mask = create_sample_data()

    # Get mask statistics
    print("Mask statistics:")
    stats = SegmentationHandler.get_mask_statistics(mask, image)
    print(f"  Voxel count: {stats['voxel_count']}")
    print(f"  Volume (mL): {stats['volume_ml']:.2f}")
    print(f"  Centroid: {stats['centroid']}")

    # Validate mask
    is_valid, message = SegmentationHandler.validate_mask(mask, image)
    print(f"\nMask validation: {message}")

    # Demonstrate morphological operations
    print("\nMorphological operations:")
    dilated = SegmentationHandler.dilate_mask(mask, radius=2)
    eroded = SegmentationHandler.erode_mask(mask, radius=2)

    dilated_stats = SegmentationHandler.get_mask_statistics(dilated)
    eroded_stats = SegmentationHandler.get_mask_statistics(eroded)

    print(f"  Original voxels: {stats['voxel_count']}")
    print(f"  Dilated voxels: {dilated_stats['voxel_count']}")
    print(f"  Eroded voxels: {eroded_stats['voxel_count']}")


def example_export():
    """Export functionality example."""
    print("\n" + "=" * 60)
    print("Example 5: Export Results")
    print("=" * 60)

    # Get features
    features = example_basic_extraction()

    # Export to different formats
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = output_dir / "features.csv"
    ResultsExporter.to_csv(features, csv_path)
    print(f"\nExported to CSV: {csv_path}")

    # JSON
    json_path = output_dir / "features.json"
    ResultsExporter.to_json(features, json_path)
    print(f"Exported to JSON: {json_path}")

    # Generate report
    print("\nGenerated report:")
    report = ResultsExporter.generate_report(features)
    print(f"  Samples: {report['summary']['n_samples']}")
    print(f"  Features: {report['summary']['n_features']}")
    print("  Categories:")
    for cat, info in report['feature_categories'].items():
        print(f"    {info['name']}: {info['count']} features")


def example_available_features():
    """List all available features."""
    print("\n" + "=" * 60)
    print("Example 6: Available Features")
    print("=" * 60)

    available = RadiomicsExtractor.get_available_features()

    print("\nAvailable feature classes:")
    for feature_class, features in available.items():
        print(f"\n{feature_class.upper()} ({len(features)} features):")
        for feat in features[:5]:
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RADIOMICS PLATFORM - EXAMPLE WORKFLOWS")
    print("=" * 60)

    # Run examples
    example_basic_extraction()
    example_custom_configuration()
    example_preprocessing()
    example_segmentation()
    example_export()
    example_available_features()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
