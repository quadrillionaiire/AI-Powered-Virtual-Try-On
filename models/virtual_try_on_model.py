import cv2
import numpy as np

def virtual_try_on(test_image_path, cloth_image_path, output_path):
    test_img = cv2.imread(test_image_path)
    cloth_img = cv2.imread(cloth_image_path)

    # Simple Overlay Example: Blend Images
    blended_img = cv2.addWeighted(test_img, 0.5, cloth_img, 0.5, 0)
    cv2.imwrite(output_path, blended_img)
