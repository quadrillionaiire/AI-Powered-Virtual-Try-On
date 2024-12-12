from utils import preprocess_image
import pytest

# Test preprocess_image
@pytest.mark.parametrize("image_path, output_path, img_size", [
    ("test_image.jpg", "output_test.jpg", (256, 192)),
])
def test_preprocess_image(image_path, output_path, img_size):
    try:
        preprocess_image(image_path, output_path, img_size)
        assert os.path.exists(output_path)
    except FileNotFoundError as e:
        print(e)


