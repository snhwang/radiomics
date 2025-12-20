# Radiomics Medical Image Analysis Platform

## Overview
A professional, user-friendly web interface for extracting and analyzing radiomics features from medical images using PyRadiomics. Built with Streamlit.

## Project Structure
```
radiomics/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
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
├── assets/
│   └── style.css             # Custom styling
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Running the Application
The application runs on Streamlit and is configured to use port 5000:
```bash
streamlit run app.py
```

## Features
- Upload medical images (NIfTI, NRRD, MetaImage, DICOM)
- Interactive image visualization with slice viewer
- Radiomics feature extraction using PyRadiomics
- Feature analysis and correlation visualization
- Export results to CSV, Excel, or JSON

## Dependencies
Key packages:
- streamlit: Web interface
- pyradiomics: Feature extraction
- SimpleITK: Medical image I/O
- plotly/matplotlib: Visualization
- pandas: Data handling

## Configuration
Streamlit is configured in `.streamlit/config.toml` for Replit environment:
- Port: 5000
- Address: 0.0.0.0
- CORS: disabled for iframe compatibility
