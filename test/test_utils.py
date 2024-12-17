from src.utils import plot_images, preprocess_image
import os
import pytest

@pytest.mark.parametrize("image_path, output_path, img_size", [
    ("tests/test_image.jpg", "tests/output_image.jpg", (256, 192)),
])
def test_preprocess_image(image_path, output_path, img_size):
    preprocess_image(image_path, output_path, img_size)
    assert os.path.exists(output_path)

