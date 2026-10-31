
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(
    "models/"
    "road_damage_yolo11n.pt"
)

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}

# Best checkpoint validation metrics
MODEL_METRICS = {
    "Precision": 46.2,
    "Recall": 30.1,
    "mAP50": 30.3,
    "mAP50-95": 12.9,
}

CLASS_METRICS = {
    "Longitudinal crack": 22.4,
    "Transverse crack": 7.16,
    "Alligator crack": 58.9,
    "Pothole": 32.8,
}


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Road Surface Intelligence",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-top: 5px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


st.markdown(
    '<p class="main-title">🛣️ Road Surface Intelligence</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'AI-powered road damage detection and analysis | YOLO11n'
    '</p>',
    unsafe_allow_html=True,
)

st.divider()

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

st.sidebar.caption(
    "Higher thresholds reduce low-confidence detections "
    "but may miss actual damage."
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")
st.sidebar.write("**Architecture:** YOLO11n")
st.sidebar.write("**Framework:** Ultralytics")
st.sidebar.write("**Dataset:** RDD2022 India")
st.sidebar.write("**Inference device:** Apple MPS")


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

st.subheader("Model Performance")

st.caption(
    "Validation results from the Experiment 2 best checkpoint."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", "46.2%")
col2.metric("Recall", "30.1%")
col3.metric("mAP@50", "30.3%")
col4.metric("mAP@50-95", "12.9%")

st.divider()


# --------------------------------------------------
# TABS
# --------------------------------------------------

analysis_tab, performance_tab, about_tab = st.tabs([
    "🔍 Image Analysis",
    "📊 Class Performance",
    "ℹ️ About Project",
])


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

with analysis_tab:

    st.subheader("Analyze Road Image")

    st.write(
        "Upload a road photograph to identify and classify "
        "visible road damage."
    )

    uploaded_files = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )


    if uploaded_files:

        st.write(f"**{len(uploaded_files)} image(s) uploaded.**")

        if st.button(
            "Analyze All Images",
            type="primary",
            width="stretch",
        ):

            all_detections = []
            processed_images = []

            progress = st.progress(0)

            for image_index, uploaded_file in enumerate(uploaded_files):

                image = Image.open(uploaded_file).convert("RGB")
                image_array = np.array(image)

                with st.spinner(f"Analyzing {uploaded_file.name}..."):

                    result = model.predict(
                        source=image_array,
                        conf=confidence,
                        imgsz=512,
                        device="mps",
                        verbose=False,
                    )[0]

                annotated = result.plot()[:, :, ::-1]
                boxes = result.boxes

                processed_images.append({
                    "filename": uploaded_file.name,
                    "original": image.copy(),
                    "annotated": Image.fromarray(annotated),
                    "count": 0 if boxes is None else len(boxes),
                })

                if boxes is not None:
                    for box in boxes:

                        class_id = int(box.cls.item())
                        score = float(box.conf.item())
                        coordinates = box.xyxy[0].tolist()

                        all_detections.append({
                            "Image": uploaded_file.name,
                            "Damage Category": CLASS_NAMES.get(
                                class_id, "Unknown"
                            ),
                            "Confidence (%)": round(score * 100, 2),
                            "X1": round(coordinates[0], 2),
                            "Y1": round(coordinates[1], 2),
                            "X2": round(coordinates[2], 2),
                            "Y2": round(coordinates[3], 2),
                        })

                progress.progress(
                    (image_index + 1) / len(uploaded_files)
                )

            st.session_state["batch_results"] = processed_images
            st.session_state["batch_detections"] = all_detections

        if "batch_results" in st.session_state:

            processed_images = st.session_state["batch_results"]
            all_detections = st.session_state["batch_detections"]

            st.divider()
            st.subheader("Batch Analysis Results")

            total_detections = len(all_detections)
            total_images = len(processed_images)

            col1, col2, col3 = st.columns(3)

            col1.metric("Images Analyzed", total_images)
            col2.metric("Total Detections", total_detections)
            col3.metric(
                "Images With Damage",
                sum(item["count"] > 0 for item in processed_images),
            )

            for item in processed_images:

                with st.expander(
                    f"{item['filename']} — {item['count']} detection(s)"
                ):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            item["original"],
                            caption="Original",
                            width="stretch",
                        )

                    with col2:
                        st.image(
                            item["annotated"],
                            caption="Detected Damage",
                            width="stretch",
                        )

            st.subheader("Combined Detection Report")

            if all_detections:

                report_df = pd.DataFrame(all_detections)

                st.dataframe(
                    report_df,
                    width="stretch",
                    hide_index=True,
                )

            else:

                report_df = pd.DataFrame(columns=[
                    "Image",
                    "Damage Category",
                    "Confidence (%)",
                    "X1", "Y1", "X2", "Y2",
                ])

                st.info("No damage was detected in the uploaded images.")

            csv_data = report_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV Report",
                data=csv_data,
                file_name="road_damage_report.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# CLASS PERFORMANCE
# --------------------------------------------------

with performance_tab:

    st.subheader("Class-wise Validation Performance")

    st.write(
        "mAP@50 for each damage category on the "
        "Experiment 2 validation dataset."
    )

    performance_df = pd.DataFrame({
        "Damage Category": list(CLASS_METRICS.keys()),
        "mAP@50 (%)": list(CLASS_METRICS.values()),
    })

    st.bar_chart(
        performance_df.set_index("Damage Category")
    )

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Transverse cracks have limited training examples. "
        "Their low validation performance is an important "
        "limitation of this model."
    )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

with about_tab:

    st.subheader("About Road Surface Intelligence")

    st.write(
        "A computer vision project designed to detect "
        "and classify road damage using deep learning."
    )

    st.markdown("""
    ### Technology Stack

    - Python
    - PyTorch
    - YOLO11n
    - OpenCV and NumPy
    - Streamlit
    - RDD2022 India Dataset

    ### Supported Damage Categories

    - Longitudinal crack
    - Transverse crack
    - Alligator crack
    - Pothole

    ### Model Details

    - Training experiment: Experiment 2
    - Training epochs: 80
    - Image size: 512 × 512
    - Inference device: Apple MPS
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Road Surface Intelligence | YOLO11n | RDD2022 India Dataset"
)

