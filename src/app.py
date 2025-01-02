import sys
import os

# Get the parent directory (AI-Powered-Virtual-Try-On)
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Append the parent directory to sys.path
sys.path.append(parent_directory)


import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import (
    live_pose_detection,
    overlay_clothing,
    build_unet
)
from utils import preprocess_image, ensure_directories
from models.virtual_try_on_model import (
     virtual_try_on_with_pose_detection,
     align_and_resize_clothing,
     get_pose_keypoints,
     load_image
 )



# Constants for Directory Paths and Image Size
RAW_DATA_PATH = "./data/raw/train"
PROCESSED_DATA_PATH = "./data/processed"
IMG_SIZE = (256, 192)

# Ensure necessary directories exist
ensure_directories()

# **Streamlit App Title**
st.title("AI-Powered Virtual Try-On")
st.markdown("""
### Revolutionizing Shopping with AI
Upload a photo or use the live webcam feature to try on clothes virtually!
""")

# **Feature 1: Live Pose Detection**
st.header("Live Pose Detection")
st.markdown("""
This feature uses your webcam to detect pose landmarks in real-time.
Press the button below to start live pose detection. Press 'q' in the webcam window to exit.
""")
if st.button("Start Live Pose Detection"):
    st.info("Opening webcam...")
    live_pose_detection()

# **Feature 2: Upload Images**
st.header("Upload Your Photo and Clothing Item")
test_image = st.file_uploader("Upload Your Photo", type=["jpg", "png"])
cloth_image = st.file_uploader("Upload Clothing Item", type=["jpg", "png"])

if test_image and cloth_image:
    # Save uploaded images to specified directories
    test_image_path = os.path.join(PROCESSED_DATA_PATH, "image", test_image.name)
    cloth_image_path = os.path.join(PROCESSED_DATA_PATH, "cloth", cloth_image.name)

    with open(test_image_path, "wb") as f:
        f.write(test_image.getbuffer())
    with open(cloth_image_path, "wb") as f:
        f.write(cloth_image.getbuffer())
    st.success("Images uploaded successfully!")

    # Preprocess uploaded images (resizing for consistency)
    processed_test_path = os.path.join(PROCESSED_DATA_PATH, "image", f"processed_{test_image.name}")
    processed_cloth_path = os.path.join(PROCESSED_DATA_PATH, "cloth", f"processed_{cloth_image.name}")
    preprocess_image(test_image_path, processed_test_path, IMG_SIZE)
    preprocess_image(cloth_image_path, processed_cloth_path, IMG_SIZE)

    # **Display Processed Images**
    st.subheader("Your Uploaded and Processed Images")
    col1, col2 = st.columns(2)
    test_img = Image.open(processed_test_path)
    cloth_img = Image.open(processed_cloth_path)
    col1.image(test_img, caption="Your Photo", use_container_width=True)
    col2.image(cloth_img, caption="Clothing Item", use_container_width=True)

    # **Virtual Try-On Feature**
    st.subheader("Virtual Try-On")
    st.markdown("This feature dynamically aligns the clothing image based on your pose landmarks.")

    # Sliders for manual adjustment (if needed)
    scaling_factor = st.slider("Adjust Scaling Factor", 0.5, 2.0, 1.0, step=0.1)
    x_offset = st.slider("Horizontal Adjustment (X)", -100, 100, 0)
    y_offset = st.slider("Vertical Adjustment (Y)", -100, 100, 0)

    # Load images for overlay
    test_image_cv = cv2.imread(processed_test_path)
    cloth_image_cv = cv2.imread(processed_cloth_path, cv2.IMREAD_UNCHANGED)

    if test_image_cv is None or cloth_image_cv is None:
        st.error("Error loading images. Please re-upload valid files.")
    else:
        # Overlay clothing with pose landmarks for alignment
        result_image = overlay_clothing(
            test_image_cv,
            cloth_image_cv,
            scaling_factor=scaling_factor,
            x_offset=x_offset,
            y_offset=y_offset,
            use_pose_landmarks=True
        )
        st.image(result_image, caption="Virtual Try-On Result", use_container_width=True)

# **Advanced Feature: Segmentation with U-Net**
st.header("Advanced: Clothing Segmentation")
st.markdown("""
Use our pre-trained U-Net model for advanced segmentation.
""")
if st.button("Run Segmentation Model"):
    st.info("Running segmentation... (Demo model)")
    unet_model = build_unet()
    # Load a dummy image and pass it through the model as a demo (to be replaced with real functionality)
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
    segmentation_result = unet_model.predict(dummy_input)
    st.image(segmentation_result[0, :, :, 0], caption="Segmentation Output", use_column_width=True)

# **End of Application**
st.markdown("""
---
Developed by [Your Name]. AI-Powered Virtual Try-On is a cutting-edge solution to enhance shopping experiences.
""")
