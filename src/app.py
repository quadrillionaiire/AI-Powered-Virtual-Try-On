import streamlit as st
from utils import preprocess_image, plot_images
from model import run_mediapipe_on_image

st.title("Virtual Try-On with Pose Detection")

# File upload widgets
pose_image_file = st.file_uploader("Upload Pose Image", type=["jpg", "png"])
clothing_image_file = st.file_uploader("Upload Clothing Image", type=["jpg", "png"])

# Overlay clothing
def overlay_clothing(pose_image_path, clothing_path):
    pose_image = Image.open(pose_image_path).convert("RGBA")
    clothing_image = Image.open(clothing_path).convert("RGBA")

    position = st.slider("Adjust Overlay Position", 0, 200, 100), st.slider("Adjust Overlay Vertical", 0, 200, 100)
    pose_image.paste(clothing_image, position, clothing_image)
    return pose_image

if st.button("Apply Clothing"):
    if pose_image_file and clothing_image_file:
        pose_image_path = f"temp_pose_image.{pose_image_file.name.split('.')[-1]}"
        clothing_image_path = f"temp_clothing_image.{clothing_image_file.name.split('.')[-1]}"

        with open(pose_image_path, "wb") as f:
            f.write(pose_image_file.read())
        with open(clothing_image_path, "wb") as f:
            f.write(clothing_image_file.read())

        output_image = overlay_clothing(pose_image_path, clothing_image_path)
        st.image(output_image, caption="Virtual Try-On Result", use_column_width=True)
    else:
        st.error("Please upload both a pose image and a clothing image.")