import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Plastic Detector", layout="wide")
st.title("🧠 Plastic Classification System (CNN + OpenCV)")
IMG_SIZE = 128  # Should match your training resolution

# -------------------------------------------------------------
# LOAD MODEL + LABEL ENCODER
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("plastic_cnn.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("plastic_cnn.h5")

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    CLASS_NAMES = list(label_encoder.classes_)
    return model, CLASS_NAMES

model, CLASS_NAMES = load_model()
st.success("✅ Model Loaded Successfully!")


# -------------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------------
def classify_image(img):
    try:
        image_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        image_array = img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        pred = model.predict(image_array)
        idx = np.argmax(pred)

        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
        conf = float(pred[0][idx] * 100)
        return label, conf

    except:
        return "Error", 0.0


# -------------------------------------------------------------
# FAKE BOUNDING BOX DETECTOR (CONTOUR BASED)
# -------------------------------------------------------------
def detect_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return frame, None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return frame, (x, y, w, h)


# -------------------------------------------------------------
# TABS: CAMERA | UPLOAD | SNAPSHOT
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📷 Real-Time Camera", "📁 Upload Image", "📸 Snapshot Capture"])


# -------------------------------------------------------------
# 1️⃣ REAL-TIME CAMERA MODE
# -------------------------------------------------------------
with tab1:
    st.subheader("📷 Live Camera Detection")

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True

    with col2:
        if st.button("⏹ Stop Camera"):
            st.session_state.camera_on = False

    frame_placeholder = st.empty()

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
            st.session_state.camera_on = False
        else:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                rgb, box = detect_region(rgb)

                if box:
                    x, y, w, h = box
                    roi = rgb[y:y+h, x:x+w]
                    label, conf = classify_image(roi)

                    cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        rgb, f"{label} ({conf:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2
                    )

                frame_placeholder.image(rgb, channels="RGB")

            cap.release()



# -------------------------------------------------------------
# 2️⃣ UPLOAD IMAGE MODE
# -------------------------------------------------------------
with tab2:
    st.subheader("📁 Upload an Image")

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file:
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Detect and classify
        img_rgb, box = detect_region(img)

        if box:
            x, y, w, h = box
            roi = img_rgb[y:y+h, x:x+w]

            label, conf = classify_image(roi)

            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{label} ({conf:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        st.image(img_rgb, caption="Detected Output", use_column_width=True)


# -------------------------------------------------------------
# 3️⃣ SNAPSHOT CAPTURE FROM CAMERA
# -------------------------------------------------------------
with tab3:
    st.subheader("📸 Capture Image from Camera")

    capture_button = st.button("Capture")

    if capture_button:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.flip(frame, 1)
            st.image(frame, caption="Captured Image", use_column_width=True)

            # Process the captured frame
            frame_rgb, box = detect_region(frame)

            if box:
                x, y, w, h = box
                roi = frame_rgb[y:y+h, x:x+w]
                label, conf = classify_image(roi)

                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{label} ({conf:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            st.image(frame_rgb, caption="Classified Image", use_column_width=True)

