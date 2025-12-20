"""
Radiomics Medical Image Analysis Platform

A professional, user-friendly web interface for extracting and analyzing
radiomics features from medical images.

Run with: streamlit run app.py
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import SimpleITK as sitk
import streamlit as st
from streamlit_option_menu import option_menu

# Import custom modules
from src.extractor import ExtractionConfig, RadiomicsExtractor
from src.visualization import RadiomicsVisualizer
from src.preprocessing import ImagePreprocessor, PreprocessingConfig, CT_WINDOW_PRESETS
from src.segmentation import SegmentationHandler
from src.export import ResultsExporter

# Page configuration
st.set_page_config(
    page_title="Radiomics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "image": None,
        "mask": None,
        "image_array": None,
        "mask_array": None,
        "features": None,
        "image_info": None,
        "mask_info": None,
        "batch_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>Radiomics Medical Image Analysis Platform</h1>
            <p>Extract quantitative imaging features from medical images for clinical research and analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
        st.title("Navigation")

        selected = option_menu(
            menu_title=None,
            options=["Upload", "Visualize", "Extract", "Analyze", "Batch", "Help"],
            icons=["cloud-upload", "image", "gear", "bar-chart", "folder", "question-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#667eea"},
            },
        )

        st.markdown("---")

        # Quick stats
        if st.session_state.image is not None:
            st.markdown("### Current Session")
            st.markdown("**Image loaded**")
            if st.session_state.mask is not None:
                st.markdown("**Mask loaded**")
            if st.session_state.features is not None:
                n_features = len(st.session_state.features.columns)
                st.markdown(f"**{n_features} features extracted**")

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8rem;'>
                Powered by PyRadiomics<br>
                v1.0.0
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected


def load_medical_image(uploaded_file) -> Tuple[Optional[sitk.Image], Optional[dict]]:
    """Load a medical image from uploaded file."""
    try:
        # Save to temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Read image
        image = sitk.ReadImage(tmp_path)

        # Get image info
        info = {
            "filename": uploaded_file.name,
            "size": image.GetSize(),
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "dimension": image.GetDimension(),
            "pixel_type": image.GetPixelIDTypeAsString(),
        }

        # Cleanup
        Path(tmp_path).unlink()

        return image, info

    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None


def render_upload_page():
    """Render the upload page."""
    st.header("Upload Medical Images")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image")
        st.markdown(
            """
            <div class="info-box">
                <strong>Supported formats:</strong> NIfTI (.nii, .nii.gz), NRRD, MetaImage (.mha, .mhd), DICOM
            </div>
            """,
            unsafe_allow_html=True,
        )

        image_file = st.file_uploader(
            "Upload medical image",
            type=["nii", "nii.gz", "nrrd", "mha", "mhd", "dcm"],
            key="image_uploader",
            help="Upload the medical image file",
        )

        if image_file:
            with st.spinner("Loading image..."):
                image, info = load_medical_image(image_file)
                if image is not None:
                    st.session_state.image = image
                    st.session_state.image_array = sitk.GetArrayFromImage(image)
                    st.session_state.image_info = info

                    st.success("Image loaded successfully!")

                    # Display image info
                    with st.expander("Image Information", expanded=True):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Dimensions", f"{info['dimension']}D")
                            st.metric("Size", f"{info['size']}")
                        with col_b:
                            st.metric("Spacing", f"{[round(s, 2) for s in info['spacing']]}")
                            st.metric("Type", info['pixel_type'])

    with col2:
        st.subheader("Segmentation Mask")
        st.markdown(
            """
            <div class="info-box">
                <strong>Note:</strong> The mask should have the same dimensions as the image.
                Use integer labels (1, 2, 3...) for different regions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        mask_file = st.file_uploader(
            "Upload segmentation mask",
            type=["nii", "nii.gz", "nrrd", "mha", "mhd"],
            key="mask_uploader",
            help="Upload the segmentation mask file",
        )

        if mask_file:
            with st.spinner("Loading mask..."):
                mask, info = load_medical_image(mask_file)
                if mask is not None:
                    mask = sitk.Cast(mask, sitk.sitkUInt8)
                    st.session_state.mask = mask
                    st.session_state.mask_array = sitk.GetArrayFromImage(mask)
                    st.session_state.mask_info = info

                    st.success("Mask loaded successfully!")

                    # Display mask info
                    mask_arr = st.session_state.mask_array
                    unique_labels = np.unique(mask_arr)
                    unique_labels = unique_labels[unique_labels > 0]

                    with st.expander("Mask Information", expanded=True):
                        st.metric("Labels found", len(unique_labels))
                        st.write(f"Label values: {list(unique_labels)}")

                        # Validate mask
                        if st.session_state.image is not None:
                            is_valid, msg = SegmentationHandler.validate_mask(
                                mask, st.session_state.image
                            )
                            if is_valid:
                                st.success(msg)
                            else:
                                st.warning(msg)

    # Quick mask creation options
    if st.session_state.image is not None and st.session_state.mask is None:
        st.markdown("---")
        st.subheader("Create Mask")

        st.markdown(
            """
            <div class="warning-box">
                No mask uploaded. You can create a simple mask using thresholding.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Threshold-based Mask Creation"):
            img_arr = st.session_state.image_array

            col1, col2 = st.columns(2)
            with col1:
                lower = st.number_input(
                    "Lower threshold",
                    value=float(np.percentile(img_arr, 25)),
                    help="Lower intensity threshold",
                )
            with col2:
                upper = st.number_input(
                    "Upper threshold",
                    value=float(np.percentile(img_arr, 75)),
                    help="Upper intensity threshold",
                )

            fill_holes = st.checkbox("Fill holes", value=True)
            largest_only = st.checkbox("Keep largest component only", value=True)

            if st.button("Create Mask", type="primary"):
                with st.spinner("Creating mask..."):
                    mask = SegmentationHandler.create_mask_from_threshold(
                        st.session_state.image,
                        lower=lower,
                        upper=upper,
                        fill_holes=fill_holes,
                        largest_component=largest_only,
                    )
                    st.session_state.mask = mask
                    st.session_state.mask_array = sitk.GetArrayFromImage(mask)
                    st.success("Mask created successfully!")
                    st.rerun()


def render_visualize_page():
    """Render the visualization page."""
    st.header("Image Visualization")

    if st.session_state.image is None:
        st.warning("Please upload an image first.")
        return

    img_arr = st.session_state.image_array
    mask_arr = st.session_state.mask_array

    # Visualization controls
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        view_axis = st.selectbox(
            "View axis",
            options=[("Axial (Z)", 0), ("Coronal (Y)", 1), ("Sagittal (X)", 2)],
            format_func=lambda x: x[0],
        )[1]

    with col2:
        max_slice = img_arr.shape[view_axis] - 1
        slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2)

    with col3:
        show_mask = st.checkbox("Show mask overlay", value=True)
        mask_opacity = st.slider("Mask opacity", 0.0, 1.0, 0.4) if show_mask else 0.4

    # Get slice
    if view_axis == 0:
        img_slice = img_arr[slice_idx, :, :]
        mask_slice = mask_arr[slice_idx, :, :] if mask_arr is not None else None
    elif view_axis == 1:
        img_slice = img_arr[:, slice_idx, :]
        mask_slice = mask_arr[:, slice_idx, :] if mask_arr is not None else None
    else:
        img_slice = img_arr[:, :, slice_idx]
        mask_slice = mask_arr[:, :, slice_idx] if mask_arr is not None else None

    # Create visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        if show_mask and mask_slice is not None and mask_slice.any():
            fig = RadiomicsVisualizer.plot_image_with_mask(
                img_arr, mask_arr, slice_idx, view_axis,
                title=f"Image with Segmentation - Slice {slice_idx}",
                mask_alpha=mask_opacity,
            )
        else:
            fig = RadiomicsVisualizer.plot_image_slice(
                img_arr, slice_idx, view_axis,
                title=f"Medical Image - Slice {slice_idx}",
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Histogram
        st.subheader("Intensity Distribution")
        fig_hist = RadiomicsVisualizer.create_histogram_distribution(
            img_slice, title="Slice Intensity Histogram"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Mask statistics
        if mask_arr is not None:
            st.subheader("Mask Statistics")
            stats = SegmentationHandler.get_mask_statistics(
                st.session_state.mask, st.session_state.image
            )
            if "error" not in stats:
                st.metric("Volume (mL)", f"{stats['volume_ml']:.2f}")
                st.metric("Voxel Count", stats['voxel_count'])

    # Multi-slice view
    with st.expander("Multi-slice View"):
        n_slices = st.slider("Number of slices", 4, 16, 9)
        fig_multi = RadiomicsVisualizer.plot_multi_slice(
            img_arr, mask_arr, n_slices, view_axis,
            title="Multi-slice Overview"
        )
        st.plotly_chart(fig_multi, use_container_width=True)


def render_extract_page():
    """Render the feature extraction page."""
    st.header("Radiomics Feature Extraction")

    if st.session_state.image is None:
        st.warning("Please upload an image first.")
        return

    if st.session_state.mask is None:
        st.warning("Please upload or create a segmentation mask first.")
        return

    # Extraction configuration
    st.subheader("Extraction Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Classes**")
        enable_firstorder = st.checkbox("First Order Statistics", value=True,
                                        help="Mean, median, variance, entropy, etc.")
        enable_shape = st.checkbox("Shape Features", value=True,
                                   help="Volume, surface area, sphericity, etc.")
        enable_glcm = st.checkbox("GLCM", value=True,
                                  help="Gray Level Co-occurrence Matrix features")
        enable_glrlm = st.checkbox("GLRLM", value=True,
                                   help="Gray Level Run Length Matrix features")
        enable_glszm = st.checkbox("GLSZM", value=True,
                                   help="Gray Level Size Zone Matrix features")
        enable_gldm = st.checkbox("GLDM", value=True,
                                  help="Gray Level Dependence Matrix features")
        enable_ngtdm = st.checkbox("NGTDM", value=True,
                                   help="Neighbouring Gray Tone Difference Matrix features")

    with col2:
        st.markdown("**Image Filters**")
        enable_wavelet = st.checkbox("Wavelet Transform", value=False,
                                     help="Extract features from wavelet decompositions")
        enable_log = st.checkbox("Laplacian of Gaussian", value=False,
                                 help="Extract features from LoG filtered images")

        if enable_log:
            log_sigma = st.multiselect(
                "LoG Sigma values",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                default=[1.0, 2.0, 3.0],
            )
        else:
            log_sigma = [1.0, 2.0, 3.0]

        st.markdown("**Other Settings**")
        bin_width = st.slider("Bin width", 5, 100, 25,
                              help="Width of bins for intensity discretization")
        normalize = st.checkbox("Normalize image", value=True)

        # Label selection
        mask_arr = st.session_state.mask_array
        unique_labels = np.unique(mask_arr)
        unique_labels = unique_labels[unique_labels > 0]
        label = st.selectbox("Label to extract", options=list(unique_labels), index=0)

    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            force_2d = st.checkbox("Force 2D extraction", value=False)
            glcm_distances = st.multiselect(
                "GLCM distances", options=[1, 2, 3, 4, 5], default=[1]
            )
        with col2:
            resample = st.checkbox("Resample to isotropic", value=False)
            if resample:
                target_spacing = st.number_input("Target spacing (mm)", value=1.0, min_value=0.1)
            else:
                target_spacing = None

    # Extract button
    st.markdown("---")

    if st.button("Extract Features", type="primary", use_container_width=True):
        # Create config
        config = ExtractionConfig(
            normalize=normalize,
            bin_width=bin_width,
            enable_first_order=enable_firstorder,
            enable_shape=enable_shape,
            enable_glcm=enable_glcm,
            enable_glrlm=enable_glrlm,
            enable_glszm=enable_glszm,
            enable_gldm=enable_gldm,
            enable_ngtdm=enable_ngtdm,
            enable_wavelet=enable_wavelet,
            enable_log=enable_log,
            log_sigma=log_sigma,
            force_2d=force_2d,
            glcm_distances=glcm_distances,
            label=int(label),
            resample_pixel_spacing=[target_spacing] * 3 if target_spacing else None,
        )

        # Extract
        with st.spinner("Extracting radiomics features..."):
            try:
                extractor = RadiomicsExtractor(config)
                features = extractor.extract_features(
                    st.session_state.image,
                    st.session_state.mask,
                    label=int(label),
                )
                st.session_state.features = features

                st.success(f"Successfully extracted {len(features.columns)} features!")

                # Display summary
                st.subheader("Extraction Summary")
                report = ResultsExporter.generate_report(features)

                cols = st.columns(len(report["feature_categories"]))
                for idx, (cat, info) in enumerate(report["feature_categories"].items()):
                    with cols[idx]:
                        st.metric(info["name"], info["count"])

            except Exception as e:
                st.error(f"Feature extraction failed: {e}")


def render_analyze_page():
    """Render the analysis page."""
    st.header("Feature Analysis")

    if st.session_state.features is None:
        st.warning("Please extract features first.")
        return

    features = st.session_state.features

    # Feature summary
    st.subheader("Feature Summary")

    # Get numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if not col.startswith(("sample_", "diagnostics"))]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Features", len(feature_cols))
    with col2:
        st.metric("Samples", len(features))
    with col3:
        # Count categories
        categories = set()
        for col in feature_cols:
            for cat in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
                if cat in col.lower():
                    categories.add(cat)
        st.metric("Categories", len(categories))
    with col4:
        # Count image types
        image_types = set()
        for col in feature_cols:
            if "original" in col.lower():
                image_types.add("Original")
            if "wavelet" in col.lower():
                image_types.add("Wavelet")
            if "log-sigma" in col.lower():
                image_types.add("LoG")
        st.metric("Image Types", len(image_types))

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Table", "Distributions", "Correlations", "Export"])

    with tab1:
        st.subheader("Feature Values")

        # Search/filter
        search = st.text_input("Search features", placeholder="Enter feature name...")

        # Filter features
        if search:
            display_cols = [col for col in feature_cols if search.lower() in col.lower()]
        else:
            display_cols = feature_cols

        # Display table
        display_df = features[display_cols].T.reset_index()
        display_df.columns = ["Feature", "Value"]
        display_df["Value"] = display_df["Value"].apply(
            lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)
        )

        st.dataframe(display_df, use_container_width=True, height=400)

    with tab2:
        st.subheader("Feature Distributions")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Category selector
            category = st.selectbox(
                "Feature Category",
                options=["All", "firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
            )

            if category == "All":
                cat_features = feature_cols
            else:
                cat_features = [col for col in feature_cols if category in col.lower()]

            selected_feature = st.selectbox("Select Feature", options=cat_features)

        with col2:
            if selected_feature:
                fig = RadiomicsVisualizer.plot_feature_distribution(
                    features, selected_feature
                )
                st.plotly_chart(fig, use_container_width=True)

        # Category overview
        st.subheader("Feature Categories Overview")
        fig_cat = RadiomicsVisualizer.plot_feature_categories(features)
        st.plotly_chart(fig_cat, use_container_width=True)

    with tab3:
        st.subheader("Feature Correlations")

        top_n = st.slider("Number of features", 10, 50, 30)

        fig_corr = RadiomicsVisualizer.plot_feature_heatmap(features, top_n=top_n)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Highly correlated features
        st.subheader("Highly Correlated Feature Pairs")
        report = ResultsExporter.generate_report(features, include_correlations=True)
        if "high_correlations" in report and report["high_correlations"]:
            corr_df = pd.DataFrame(report["high_correlations"])
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("No highly correlated feature pairs found (|r| > 0.9)")

    with tab4:
        st.subheader("Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### CSV Export")
            csv_data = ResultsExporter.to_csv_bytes(features)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="radiomics_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.markdown("### Excel Export")
            excel_data = ResultsExporter.to_excel_bytes(features)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="radiomics_features.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col3:
            st.markdown("### JSON Export")
            json_data = ResultsExporter.to_json_bytes(features)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="radiomics_features.json",
                mime="application/json",
                use_container_width=True,
            )


def render_batch_page():
    """Render the batch processing page."""
    st.header("Batch Processing")

    st.markdown(
        """
        <div class="info-box">
            <strong>Batch Processing:</strong> Extract radiomics features from multiple image-mask pairs at once.
            Upload a folder containing matched image and mask files.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File upload
    st.subheader("Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Images**")
        image_files = st.file_uploader(
            "Upload images",
            type=["nii", "nii.gz", "nrrd", "mha", "mhd"],
            accept_multiple_files=True,
            key="batch_images",
        )

    with col2:
        st.markdown("**Masks**")
        mask_files = st.file_uploader(
            "Upload masks",
            type=["nii", "nii.gz", "nrrd", "mha", "mhd"],
            accept_multiple_files=True,
            key="batch_masks",
        )

    if image_files and mask_files:
        st.markdown("---")

        # Match files
        st.subheader("File Matching")

        # Sort files
        image_names = sorted([f.name for f in image_files])
        mask_names = sorted([f.name for f in mask_files])

        # Create mapping table
        n_pairs = min(len(image_files), len(mask_files))
        st.info(f"Found {n_pairs} image-mask pairs")

        # Display pairs
        pairs_df = pd.DataFrame({
            "Image": image_names[:n_pairs],
            "Mask": mask_names[:n_pairs],
        })
        st.dataframe(pairs_df, use_container_width=True)

        # Extraction settings (simplified)
        st.subheader("Extraction Settings")

        col1, col2 = st.columns(2)
        with col1:
            bin_width = st.slider("Bin width", 5, 100, 25, key="batch_bin")
            normalize = st.checkbox("Normalize", value=True, key="batch_norm")
        with col2:
            label = st.number_input("Label value", value=1, min_value=1, key="batch_label")

        # Process button
        if st.button("Process Batch", type="primary", use_container_width=True):
            config = ExtractionConfig(
                normalize=normalize,
                bin_width=bin_width,
                label=int(label),
            )
            extractor = RadiomicsExtractor(config)

            # Progress bar
            progress = st.progress(0)
            status = st.empty()

            all_features = []

            for idx, (img_file, mask_file) in enumerate(zip(image_files[:n_pairs], mask_files[:n_pairs])):
                status.text(f"Processing: {img_file.name}")

                try:
                    # Load files
                    img, _ = load_medical_image(img_file)
                    mask, _ = load_medical_image(mask_file)

                    if img is not None and mask is not None:
                        mask = sitk.Cast(mask, sitk.sitkUInt8)
                        features = extractor.extract_features(img, mask, label=int(label))
                        features["image_file"] = img_file.name
                        features["mask_file"] = mask_file.name
                        all_features.append(features)

                except Exception as e:
                    st.warning(f"Failed to process {img_file.name}: {e}")

                progress.progress((idx + 1) / n_pairs)

            if all_features:
                batch_results = pd.concat(all_features, ignore_index=True)
                st.session_state.batch_results = batch_results

                st.success(f"Successfully processed {len(all_features)} image pairs!")

                # Display results
                st.subheader("Batch Results")
                st.dataframe(batch_results.head(10), use_container_width=True)

                # Export
                st.subheader("Export Batch Results")
                col1, col2 = st.columns(2)

                with col1:
                    csv_data = ResultsExporter.to_csv_bytes(batch_results)
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="batch_radiomics_features.csv",
                        mime="text/csv",
                    )

                with col2:
                    excel_data = ResultsExporter.to_excel_bytes(batch_results)
                    st.download_button(
                        "Download Excel",
                        data=excel_data,
                        file_name="batch_radiomics_features.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )


def render_help_page():
    """Render the help page."""
    st.header("Help & Documentation")

    st.markdown(
        """
        ## Getting Started

        1. **Upload**: Start by uploading a medical image and its corresponding segmentation mask
        2. **Visualize**: Review the image and mask to ensure they're correctly loaded
        3. **Extract**: Configure extraction settings and extract radiomics features
        4. **Analyze**: Explore the extracted features through visualizations
        5. **Export**: Download results in CSV, Excel, or JSON format

        ---

        ## Supported Formats

        | Format | Extension | Description |
        |--------|-----------|-------------|
        | NIfTI | .nii, .nii.gz | Neuroimaging Informatics Technology Initiative |
        | NRRD | .nrrd | Nearly Raw Raster Data |
        | MetaImage | .mha, .mhd | ITK MetaImage format |
        | DICOM | .dcm | Digital Imaging and Communications in Medicine |

        ---

        ## Feature Classes

        ### First Order Statistics
        Describes the distribution of voxel intensities within the ROI:
        - Mean, Median, Range
        - Variance, Standard Deviation
        - Skewness, Kurtosis
        - Energy, Entropy

        ### Shape Features
        Describes the 3D size and shape of the ROI:
        - Volume, Surface Area
        - Sphericity, Compactness
        - Elongation, Flatness

        ### Texture Features

        **GLCM (Gray Level Co-occurrence Matrix)**
        - Contrast, Correlation
        - Energy, Homogeneity
        - Joint entropy

        **GLRLM (Gray Level Run Length Matrix)**
        - Run emphasis (short/long)
        - Gray level variance
        - Run length variance

        **GLSZM (Gray Level Size Zone Matrix)**
        - Zone emphasis (small/large)
        - Gray level variance
        - Zone variance

        **GLDM (Gray Level Dependence Matrix)**
        - Dependence emphasis
        - Gray level variance
        - Dependence variance

        **NGTDM (Neighbouring Gray Tone Difference Matrix)**
        - Coarseness, Contrast
        - Busyness, Complexity
        - Strength

        ---

        ## Tips

        - **Mask validation**: Ensure your mask has the same dimensions as the image
        - **Label selection**: If your mask has multiple labels, select the correct one
        - **Batch processing**: Name files consistently for automatic matching
        - **Performance**: Large images with wavelet features may take longer

        ---

        ## References

        - [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
        - [IBSI (Image Biomarker Standardisation Initiative)](https://theibsi.github.io/)
        """
    )


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    selected_page = render_sidebar()

    # Route to selected page
    if selected_page == "Upload":
        render_upload_page()
    elif selected_page == "Visualize":
        render_visualize_page()
    elif selected_page == "Extract":
        render_extract_page()
    elif selected_page == "Analyze":
        render_analyze_page()
    elif selected_page == "Batch":
        render_batch_page()
    elif selected_page == "Help":
        render_help_page()


if __name__ == "__main__":
    main()
