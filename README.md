# Radiomics Medical Image Analysis Platform

A professional, user-friendly interface for extracting and analyzing radiomics features from medical images using PyRadiomics.

## Features

- **Intuitive Web Interface**: Clean, modern Streamlit-based UI for easy interaction
- **Comprehensive Feature Extraction**: Support for all PyRadiomics feature classes
- **Interactive Visualization**: Real-time image viewing with segmentation overlay
- **Batch Processing**: Process multiple image-mask pairs efficiently
- **Multiple Export Formats**: Export results to CSV, Excel, or JSON
- **Preprocessing Pipeline**: Built-in normalization, resampling, and filtering

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/radiomics.git
cd radiomics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Supported Image Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| NIfTI | .nii, .nii.gz | Neuroimaging Informatics Technology Initiative |
| NRRD | .nrrd | Nearly Raw Raster Data |
| MetaImage | .mha, .mhd | ITK MetaImage format |
| DICOM | .dcm | Digital Imaging and Communications in Medicine |

## Feature Classes

### First Order Statistics
Describes the distribution of voxel intensities:
- Mean, Median, Range, Variance
- Skewness, Kurtosis
- Energy, Entropy

### Shape Features
Describes 3D geometry of the ROI:
- Volume, Surface Area
- Sphericity, Compactness
- Elongation, Flatness

### Texture Features

| Feature Class | Description |
|--------------|-------------|
| **GLCM** | Gray Level Co-occurrence Matrix |
| **GLRLM** | Gray Level Run Length Matrix |
| **GLSZM** | Gray Level Size Zone Matrix |
| **GLDM** | Gray Level Dependence Matrix |
| **NGTDM** | Neighbouring Gray Tone Difference Matrix |

### Image Filters
- **Wavelet Transform**: Multi-resolution decomposition
- **Laplacian of Gaussian (LoG)**: Edge enhancement at multiple scales

## Programmatic Usage

```python
from src.extractor import RadiomicsExtractor, ExtractionConfig
import SimpleITK as sitk

# Load image and mask
image = sitk.ReadImage("path/to/image.nii.gz")
mask = sitk.ReadImage("path/to/mask.nii.gz")

# Configure extraction
config = ExtractionConfig(
    normalize=True,
    bin_width=25,
    enable_first_order=True,
    enable_shape=True,
    enable_glcm=True,
    enable_wavelet=False,
)

# Extract features
extractor = RadiomicsExtractor(config)
features = extractor.extract_features(image, mask)

# Export results
from src.export import ResultsExporter
ResultsExporter.to_csv(features, "radiomics_features.csv")
```

## Project Structure

```
radiomics/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── extractor.py          # Feature extraction engine
│   ├── visualization.py      # Plotting and visualization
│   ├── preprocessing.py      # Image preprocessing
│   ├── segmentation.py       # Segmentation utilities
│   └── export.py             # Results export
├── config/
│   └── default_params.yaml   # Default extraction parameters
├── examples/
│   └── sample_workflow.py    # Example usage script
└── assets/
    └── style.css             # Custom styling
```

## Workflow

1. **Upload**: Load medical image and segmentation mask
2. **Visualize**: Review images with interactive slice viewer
3. **Configure**: Select feature classes and preprocessing options
4. **Extract**: Run radiomics feature extraction
5. **Analyze**: Explore features through visualizations
6. **Export**: Download results in preferred format

## Configuration Options

### Extraction Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `normalize` | Normalize image intensities | True |
| `bin_width` | Bin width for discretization | 25 |
| `force_2d` | Force 2D extraction | False |
| `label` | Label value in mask | 1 |

### Preprocessing Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resample` | Resample to isotropic spacing | False |
| `smoothing` | Apply Gaussian smoothing | False |
| `bias_correction` | N4 bias field correction | False |
| `windowing` | CT window/level | None |

## Dependencies

- Python 3.8+
- PyRadiomics >= 3.0.1
- SimpleITK >= 2.3.0
- Streamlit >= 1.28.0
- NumPy, Pandas, Plotly
- See `requirements.txt` for complete list

## References

- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
- [Image Biomarker Standardisation Initiative (IBSI)](https://theibsi.github.io/)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)

## License

This project is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
