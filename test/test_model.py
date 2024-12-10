from src.utils import detect_pose

# Test the pose detection with a sample image
image_path = "data/raw/images/sample_image.jpg"  # Add test image here
output_path = "data/processed/pose_output.jpg"
detect_pose(image_path, output_path)

