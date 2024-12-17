import streamlit as st
from model import live_pose_detection
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import preprocess_image, plot_images, ensure_directories



# Constants
RAW_DATA_PATH = "/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/raw/train"
PROCESSED_DATA_PATH = "/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/processed"
IMG_SIZE = (256, 192)

# Ensure necessary directories exist
ensure_directories()

# Streamlit App
st.title("AI-Powered Virtual Try-On")

# Image Uploads
st.header("Upload Test Image and Cloth")
test_image = st.file_uploader("Upload Test Image", type=["jpg", "png"])
cloth_image = st.file_uploader("Upload Cloth Image", type=["jpg", "png"])

if test_image and cloth_image:
    # Save uploaded files
    test_image_path = os.path.join(PROCESSED_DATA_PATH, "image", test_image.name)
    cloth_image_path = os.path.join(PROCESSED_DATA_PATH, "cloth", cloth_image.name)

    with open(test_image_path, "wb") as f:
        f.write(test_image.getbuffer())
    with open(cloth_image_path, "wb") as f:
        f.write(cloth_image.getbuffer())

    st.success("Images uploaded successfully!")

    # Preprocess Images
    processed_test_path = os.path.join(PROCESSED_DATA_PATH, "image", f"processed_{test_image.name}")
    processed_cloth_path = os.path.join(PROCESSED_DATA_PATH, "cloth", f"processed_{cloth_image.name}")

    preprocess_image(test_image_path, processed_test_path, IMG_SIZE)
    preprocess_image(cloth_image_path, processed_cloth_path, IMG_SIZE)

    # Display Images
    st.subheader("Processed Images")
    col1, col2 = st.columns(2)

    # Display Test Image
    test_img = Image.open(processed_test_path)
    col1.image(test_img, caption="Processed Test Image", use_column_width=True)

    # Display Cloth Image
    cloth_img = Image.open(processed_cloth_path)
    col2.image(cloth_img, caption="Processed Cloth Image", use_column_width=True)

    # Visualize Images
    st.subheader("Visualization")
    plot_images([processed_test_path, processed_cloth_path], ["Test Image", "Cloth Image"])
else:
    st.warning("Please upload both Test Image and Cloth Image.")
