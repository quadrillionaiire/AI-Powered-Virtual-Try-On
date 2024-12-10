# Pose Detection with Clothing Alignment
import cv2
import numpy as np
from PIL import Image

def overlay_clothing(pose_image_path, clothing_path, output_path):
    pose_image = Image.open(pose_image_path)
    clothing_image = Image.open(clothing_path).convert("RGBA")
    
    # Example overlay position (customize as needed)
    position = (100, 100)
    pose_image.paste(clothing_image, position, clothing_image)
    pose_image.save(output_path)
    print(f"Overlay saved to {output_path}")

