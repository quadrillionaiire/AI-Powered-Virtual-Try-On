from src.utils import preprocess_image, ensure_directories
import os

def test_ensure_directories():
    ensure_directories()
    assert os.path.exists("/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/processed/image")
    assert os.path.exists("/Users/quadrillionaiire/Documents/Phase-5-Capstone/AI-Powered-Virtual-Try-On/data/processed/cloth")

def test_preprocess_image():
    test_image_path = "tests/test_image.jpg"
    output_path = "tests/output_test.jpg"
    img_size = (256, 192)

    # Create a dummy image for testing
    import numpy as np
    import cv2
    dummy_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.imwrite(test_image_path, dummy_image)

    preprocess_image(test_image_path, output_path, img_size)
    assert os.path.exists(output_path)

