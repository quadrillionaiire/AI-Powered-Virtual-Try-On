import numpy as np
import cv2
from src.model import overlay_clothing

def test_overlay_clothing():
    # Create dummy images
    base_image = np.ones((256, 256, 3), dtype=np.uint8) * 255  # White background
    clothing = np.ones((50, 50, 3), dtype=np.uint8) * 127      # Gray clothing patch

    # Call overlay_clothing
    result = overlay_clothing(base_image, clothing, scaling_factor=2.0, x_offset=10, y_offset=10)
    
    # Check if the overlay area is modified (i.e., not all white in the overlay region)
    assert not np.all(result[10:110, 10:110] == 255), "Overlay did not apply correctly."




