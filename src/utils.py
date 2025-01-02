import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Constants for Directory Paths and Image Size
RAW_DATA_PATH = "./data/raw/train"
PROCESSED_DATA_PATH = "./data/processed"
IMG_SIZE = (256, 192)

def ensure_directories():
    """
    Ensure that the necessary directories exist for saving processed images.
    Creates 'image' and 'cloth' subdirectories within the processed data path.
    """
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "image"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, "cloth"), exist_ok=True)

def preprocess_image(image_path, output_path, img_size):
    """
    Preprocess an image by resizing and normalizing it.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the processed image.
    - img_size: Tuple (width, height) specifying the target size.

    Raises:
    - FileNotFoundError: If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load {image_path}")

    # Resize and normalize the image
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, img_resized)

def normalize_image(image):
    """
    Normalize an image by scaling pixel values to the range [0, 1].

    Parameters:
    - image: numpy.ndarray, the image to normalize.

    Returns:
    - normalized_image: numpy.ndarray, normalized image.
    """
    return image / 255.0

def denormalize_image(image):
    """
    Denormalize an image by scaling pixel values back to the range [0, 255].

    Parameters:
    - image: numpy.ndarray, the normalized image to denormalize.

    Returns:
    - denormalized_image: numpy.ndarray, denormalized image.
    """
    return (image * 255).astype(np.uint8)

def plot_images(image_paths, titles, ncols=3):
    """
    Visualize multiple images with their titles.

    Parameters:
    - image_paths: List of file paths to images.
    - titles: List of titles corresponding to each image.
    - ncols: Number of columns in the grid layout.

    Notes:
    - Automatically determines the number of rows based on the number of images.
    """
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

def batch_preprocess_images(image_dir, output_dir, img_size):
    """
    Batch preprocess all images in a directory.

    Parameters:
    - image_dir: Directory containing input images.
    - output_dir: Directory to save processed images.
    - img_size: Tuple (width, height) specifying the target size.

    Notes:
    - Ensures output directory exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(image_dir), desc="Processing images"):
        input_path = os.path.join(image_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        try:
            preprocess_image(input_path, output_path, img_size)
        except FileNotFoundError:
            print(f"Warning: Could not process {input_path}. File not found.")
