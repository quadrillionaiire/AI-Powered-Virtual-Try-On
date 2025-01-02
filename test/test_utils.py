import os
import pytest
from src.utils import preprocess_image, ensure_directories

@pytest.fixture
def test_image_paths(tmp_path):
    # Temporary input and output paths
    input_path = tmp_path / "test_input.jpg"
    output_path = tmp_path / "output.png"
    
    # Create a dummy image
    import numpy as np
    import cv2
    img = np.ones((256, 192, 3), dtype=np.uint8) * 255  # White image
    cv2.imwrite(str(input_path), img)
    
    return input_path, output_path

def test_preprocess_image(test_image_paths):
    input_path, output_path = test_image_paths
    
    # provided paths for testing
    preprocess_image(str(input_path), str(output_path), (256, 192))
    
    # Assert output file is created
    assert os.path.exists(output_path), "Output file was not created."

def test_ensure_directories(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
    ensure_directories()
    assert os.path.exists(tmp_path), "Directories were not created."
