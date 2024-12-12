import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from PIL import Image

# Define file paths
RAW_DATA_PATH = "/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/raw/train"
PROCESSED_DATA_PATH = "/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/processed"
IMG_SIZE = (256, 192)

# Ensure directories exist
def ensure_directories():
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "image"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "cloth"), exist_ok=True)

# Preprocess image
def preprocess_image(image_path, output_path, img_size):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load {image_path}")

    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

# Visualize images
def plot_images(image_paths, titles, ncols=3):
    nrows = int(np.ceil(len(image_paths) / ncols))
    plt.figure(figsize=(15, 5 * nrows))
    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        img = cv2.imread(img_path)
        if img is not None:
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
    plt.show()