import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import gdown
import os
import pandas as pd
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="License Plate Detection & OCR",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚗 License Plate Detection & OCR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect license plates with YOLOv8 and extract plate text using EasyOCR.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    try:

        # Download weights if not present
        if not os.path.exists("weights/best.pt"):

            os.makedirs(
                "weights",
                exist_ok=True
            )

            file_id = "10-ts5J9Y-7BN_K36A9kZb5rKnwCUwCed"

            url = (
                "https://drive.google.com/file/d/"
                f"{file_id}"
                "/view?usp=drive_link"
            )

            gdown.download(
                url,
                "weights/best.pt",
                quiet=False
            )

        # Load YOLO model
        model = YOLO(
            "weights/best.pt"
        )

        # Load EasyOCR
        reader = easyocr.Reader(
            ["en"]
        )

        return model, reader

    except Exception as e:

        st.error(
            f"❌ Failed to load AI models: {e}"
        )

        st.stop()


model, reader = load_models()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.markdown(
        """
        **Detection:** YOLOv8  
        **OCR:** EasyOCR  
        **Preprocessing:** OpenCV  
        **Interface:** Streamlit
        """
    )

    st.divider()

    st.markdown("### 📌 Pipeline")

    st.markdown(
        """
        1. Upload image
        2. Detect license plate
        3. Crop plate
        4. Preprocess image
        5. Read text with OCR
        6. Display confidence
        7. Export results
        """
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "📁 Upload a vehicle image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a clear image containing one or more license plates."
)


# ============================================================
# NO IMAGE
# ============================================================

if not uploaded:

    st.info(
        "👆 Upload a vehicle image above to start detection."
    )

    st.markdown("### How it works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**1️⃣ Upload**")
        st.caption("Provide a vehicle image.")

    with col2:
        st.markdown("**2️⃣ Detect**")
        st.caption("YOLOv8 finds license plates.")

    with col3:
        st.markdown("**3️⃣ Preprocess**")
        st.caption("OpenCV prepares the plate for OCR.")

    with col4:
        st.markdown("**4️⃣ Read**")
        st.caption("EasyOCR extracts the plate text.")

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    image = np.array(
        Image.open(uploaded).convert("RGB")
    )

except Exception as e:

    st.error(
        f"❌ Unable to read the uploaded image: {e}"
    )

    st.stop()


# ============================================================
# RUN YOLO DETECTION
# ============================================================

with st.spinner("🔍 Detecting license plates..."):

    try:

        results = model(image)

    except Exception as e:

        st.error(
            f"❌ Detection failed: {e}"
        )

        st.stop()


# ============================================================
# ANNOTATED IMAGE
# ============================================================

try:

    annotated = results[0].plot(
        font_size=10,
        line_width=2
    )

except Exception as e:

    st.error(
        f"❌ Unable to create detection visualization: {e}"
    )

    st.stop()


# ============================================================
# DETECTION SUMMARY
# ============================================================

boxes = results[0].boxes

number_of_plates = len(boxes)


if number_of_plates > 0:

    st.success(
        f"✅ Detected {number_of_plates} "
        f"license plate(s)."
    )

else:

    st.warning(
        "⚠️ No license plates were detected in this image."
    )


# ============================================================
# DISPLAY INPUT / DETECTION
# ============================================================

st.subheader("🖼️ Detection Results")

col1, col2 = st.columns(2)

with col1:

    st.markdown("**Original Image**")

    st.image(
        image,
        use_container_width=True
    )


with col2:

    st.markdown("**YOLOv8 Detection**")

    st.image(
        annotated,
        use_container_width=True
    )


# ============================================================
# OCR PROCESSING
# ============================================================

ocr_results = []


if number_of_plates > 0:

    st.subheader("📋 License Plate Results")

    for i, box in enumerate(boxes):

        try:

            # ------------------------------------------------
            # GET BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # KEEP COORDINATES INSIDE IMAGE
            # ------------------------------------------------

            height, width = image.shape[:2]

            x1 = max(
                0,
                min(x1, width)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y1 = max(
                0,
                min(y1, height)
            )

            y2 = max(
                0,
                min(y2, height)
            )


            # ------------------------------------------------
            # YOLO CONFIDENCE
            # ------------------------------------------------

            yolo_confidence = (
                float(box.conf[0]) * 100
            )


            # ------------------------------------------------
            # CROP PLATE
            # ------------------------------------------------

            plate = image[
                y1:y2,
                x1:x2
            ]


            if plate.size == 0:

                continue


            # =================================================
            # OCR PREPROCESSING
            # =================================================

            # Resize
            plate_resized = cv2.resize(
                plate,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )


            # RGB → grayscale
            gray = cv2.cvtColor(
                plate_resized,
                cv2.COLOR_RGB2GRAY
            )


            # Reduce small noise
            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )


            # Otsu thresholding
            thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]


            # =================================================
            # EASY OCR
            # =================================================

            ocr_result = reader.readtext(
                thresh,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                detail=1
            )


            # ------------------------------------------------
            # OCR RESULT PROCESSING
            # ------------------------------------------------

            detected_text = ""
            ocr_confidence = 0.0


            if ocr_result:

                # Sort OCR detections from left to right
                ocr_result = sorted(
                    ocr_result,
                    key=lambda item: min(
                        point[0]
                        for point in item[0]
                    )
                )


                # Combine detected text
                detected_text = "".join(
                    item[1]
                    for item in ocr_result
                )


                # Average OCR confidence
                confidence_values = [
                    float(item[2])
                    for item in ocr_result
                ]


                if confidence_values:

                    ocr_confidence = (
                        sum(confidence_values)
                        / len(confidence_values)
                    ) * 100


            # ------------------------------------------------
            # CLEAN OCR TEXT
            # ------------------------------------------------

            detected_text = detected_text.upper()

            detected_text = re.sub(
                r"[^A-Z0-9-]",
                "",
                detected_text
            )


            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            if detected_text:

                ocr_results.append(
                    {
                        "Plate": f"Plate {i + 1}",
                        "Detected Text": detected_text,
                        "YOLO Confidence (%)": round(
                            yolo_confidence,
                            1
                        ),
                        "OCR Confidence (%)": round(
                            ocr_confidence,
                            1
                        ),
                        "Coordinates": (
                            f"({x1}, {y1}) - "
                            f"({x2}, {y2})"
                        )
                    }
                )

            else:

                ocr_results.append(
                    {
                        "Plate": f"Plate {i + 1}",
                        "Detected Text": "Not detected",
                        "YOLO Confidence (%)": round(
                            yolo_confidence,
                            1
                        ),
                        "OCR Confidence (%)": 0.0,
                        "Coordinates": (
                            f"({x1}, {y1}) - "
                            f"({x2}, {y2})"
                        )
                    }
                )


            # =================================================
            # DISPLAY INDIVIDUAL PLATE
            # =================================================

            st.markdown(
                f"### 🚘 Plate {i + 1}"
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.markdown("**Cropped Plate**")

                st.image(
                    plate,
                    use_container_width=True
                )


            with result_col2:

                st.markdown("**Processed for OCR**")

                st.image(
                    thresh,
                    use_container_width=True
                )


            # ------------------------------------------------
            # CONFIDENCE METRICS
            # ------------------------------------------------

            metric1, metric2, metric3 = st.columns(3)


            with metric1:

                st.metric(
                    "YOLO Confidence",
                    f"{yolo_confidence:.1f}%"
                )


            with metric2:

                st.metric(
                    "OCR Confidence",
                    f"{ocr_confidence:.1f}%"
                )


            with metric3:

                if detected_text:

                    st.metric(
                        "Characters",
                        len(detected_text)
                    )

                else:

                    st.metric(
                        "Characters",
                        "—"
                    )


            # ------------------------------------------------
            # TEXT RESULT
            # ------------------------------------------------

            if detected_text:

                st.success(
                    f"🔤 Detected Plate: **{detected_text}**"
                )

            else:

                st.warning(
                    "⚠️ OCR could not confidently extract "
                    "plate text from this crop."
                )


            st.divider()


        except Exception as e:

            st.error(
                f"❌ Error processing Plate {i + 1}: {e}"
            )


# ============================================================
# RESULTS TABLE
# ============================================================

if ocr_results:

    st.subheader("📊 Detection & OCR Summary")

    results_df = pd.DataFrame(
        ocr_results
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPORT RESULTS
    # ========================================================

    st.subheader("📥 Export Results")

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_data,
        file_name="license_plate_results.csv",
        mime="text/csv"
    )


else:

    if number_of_plates > 0:

        st.warning(
            "⚠️ License plates were detected, "
            "but no readable OCR text was extracted."
        )

    else:

        st.info(
            "No license plate results available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "License Plate Detection & OCR | "
    "YOLOv8 + OpenCV + EasyOCR + Streamlit"
)
