import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# Page config
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 License Plate Detection & OCR")
st.markdown("Upload an image to **detect** and **read** the license plate automatically.")

# Load models (cached so they don't reload every time)
@st.cache_resource


@st.cache_resource
def load_models():
    # Download weights if not present
    if not os.path.exists('weights/best.pt'):
        os.makedirs('weights', exist_ok=True)
        # Convert share link to direct download link
        file_id = '10-ts5J9Y-7BN_K36A9kZb5rKnwCUwCed'
        url = f'https://drive.google.com/file/d/10-ts5J9Y-7BN_K36A9kZb5rKnwCUwCed/view?usp=drive_link'
        gdown.download(url, 'weights/best.pt', quiet=False)

    model = YOLO('weights/best.pt')
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

# Upload image
uploaded = st.file_uploader("📁 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    # Convert to numpy array
    image = np.array(Image.open(uploaded))

    # Run detection
    with st.spinner("Detecting license plate..."):
        results = model(image)
        annotated = results[0].plot(font_size=10, line_width=1)

    # Show input and output side by side
    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Image**")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("**Detected Plate**")
        st.image(annotated, use_column_width=True)

    # OCR on each detected plate
    st.subheader("📋 Plate Text")

    plate_found = False

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # Preprocess
        plate_resized = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR
        ocr_result = reader.readtext(
            thresh,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        )

        if ocr_result:
            plate_found = True
            for detection in ocr_result:
                text = detection[1]
                confidence = round(detection[2] * 100, 1)
                st.success(f"🔤 Plate {i+1}: **{text}** — Confidence: {confidence}%")

                # Show cropped plate
                st.image(thresh, caption=f"Processed Plate {i+1}", width=300)

    if not plate_found:
        st.warning("No plate text found. Try a clearer image.")

else:
    # Show instructions when no image uploaded
    st.info("👆 Upload a car image above to get started.")
    st.markdown("""
    ### How it works
    1. Upload any car image
    2. YOLOv8 detects the license plate
    3. EasyOCR reads the plate text
    4. Results shown instantly
    """)