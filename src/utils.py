import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

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

    # Convert BGR to RGB to fix color issues
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_AREA)
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

# Load pre-trained segmentation model
def load_segmentation_model():
    """
    Loads the pre-trained segmentation model.
    Returns:
    - model: A pre-trained DeepLabV3 ResNet50 model.
    """
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model

# Segment cloth area
def segment_cloth_area(model, cloth_image_path):
    """
    Segments the cloth area from an input image using a segmentation model.

    Parameters:
    - model: A pre-trained segmentation model.
    - cloth_image_path: Path to the cloth image to be segmented.

    Returns:
    - img_rgba: The image with a transparent background for non-cloth areas (RGBA format).
    - mask: The binary segmentation mask identifying the cloth area.
    """
    # Load and preprocess the image
    img = Image.open(cloth_image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    # Perform segmentation
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().numpy()

    # Create an RGBA image with transparent background
    img_rgba = np.array(img.convert("RGBA"))
    img_rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255
    return img_rgba, mask

# Visualize images
def visualize_images(image_paths, cloth_paths, output_paths):
    def update(val):
        idx = int(slider.val)

        # Ensure proper loading of images
        img_ori = mpimg.imread(image_paths[idx])
        cloth = mpimg.imread(cloth_paths[idx])
        img_new = mpimg.imread(output_paths[idx], format="png")  # Ensure correct format

        # Clear axes to avoid overlaps
        for ax in axes:
            ax.clear()

        # Display images with correct channels
        axes[0].imshow(img_ori)
        axes[0].axis("off")
        axes[0].set_title("Original")

        axes[1].imshow(cloth)
        axes[1].axis("off")
        axes[1].set_title("Cloth")

        axes[2].imshow(img_new, cmap='gray')  # Use cmap='gray' for consistent grayscale visualization
        axes[2].axis("off")
        axes[2].set_title("Generated")

        fig.canvas.draw_idle()  # Redraw canvas after updates

    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    # Add slider for interactivity
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, "Image Index", 0, len(image_paths) - 1, valinit=0, valstep=1)
    slider.on_changed(update)

    update(0)  # Initial visualization
    plt.show()

# Example usage
if __name__ == "__main__":
    ensure_directories()
    image_paths = [os.path.join(PROCESSED_DATA_PATH, "image", f"image{i}.jpg") for i in range(1, 4)]
    cloth_paths = [os.path.join(PROCESSED_DATA_PATH, "cloth", f"cloth{i}.jpg") for i in range(1, 4)]
    output_paths = [os.path.join(PROCESSED_DATA_PATH, "output", f"output{i}.png") for i in range(1, 4)]

    visualize_images(image_paths, cloth_paths, output_paths)
