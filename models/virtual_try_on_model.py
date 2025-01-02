import os
import cv2
import numpy as np
import mediapipe as mp
from src.utils import segment_cloth_area, load_segmentation_model

# Load MediaPipe Pose once to save initialization time
mp_pose = mp.solutions.pose

def load_image(image_input):
    """
    Load an image from a file path or directly use a numpy array.

    Parameters:
    - image_input: str (file path) or numpy.ndarray (image)

    Returns:
    - image: numpy.ndarray
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image at {image_input} not found.")
        image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("Invalid image_input type. Must be a file path or a numpy array.")
    if image is None:
        raise ValueError(f"Failed to read image: {image_input}")
    return image

def get_pose_keypoints(image):
    """
    Extract pose keypoints from an image using MediaPipe.

    Parameters:
    - image: numpy.ndarray (RGB or BGR)

    Returns:
    - keypoints: List of (x, y) tuples for pose landmarks
    """
    with mp_pose.Pose() as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError("Pose landmarks not detected.")

        keypoints = [
            (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
            for landmark in results.pose_landmarks.landmark
        ]
    return keypoints

def align_and_resize_clothing(clothing_img, left_shoulder, right_shoulder):
    """
    Resize and align clothing image based on shoulder keypoints.

    Parameters:
    - clothing_img: numpy.ndarray (clothing image)
    - left_shoulder, right_shoulder: (x, y) tuples for shoulder points

    Returns:
    - resized_clothing: numpy.ndarray (aligned and resized clothing image)
    - offset: (x_offset, y_offset) for placement
    """
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                       (left_shoulder[1] + right_shoulder[1]) // 2)
    shoulder_distance = int(np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)))

    # Scale clothing image based on shoulder width
    scaling_factor = shoulder_distance / clothing_img.shape[1]
    resized_clothing = cv2.resize(clothing_img, None, fx=scaling_factor, fy=scaling_factor)

    # Calculate offsets for alignment
    x_offset = max(0, shoulder_center[0] - resized_clothing.shape[1] // 2)
    y_offset = max(0, shoulder_center[1] - resized_clothing.shape[0] // 2)

    return resized_clothing, (x_offset, y_offset)

def overlay_clothing(user_img, clothing_img, offset):
    """
    Overlay clothing image onto user image at the specified offset.

    Parameters:
    - user_img: numpy.ndarray (user image)
    - clothing_img: numpy.ndarray (clothing image with alpha channel)
    - offset: (x_offset, y_offset) tuple for placement

    Returns:
    - overlaid_image: numpy.ndarray
    """
    x_offset, y_offset = offset
    h, w = clothing_img.shape[:2]

    # Ensure the clothing fits within the user image boundaries
    x_end = min(x_offset + w, user_img.shape[1])
    y_end = min(y_offset + h, user_img.shape[0])
    cropped_clothing = clothing_img[:y_end - y_offset, :x_end - x_offset]

    # Extract alpha channel for blending
    alpha_clothing = cropped_clothing[:, :, 3] / 255.0  # Normalize alpha values
    for c in range(3):  # Apply blending to R, G, B channels
        user_img[y_offset:y_end, x_offset:x_end, c] = (
            alpha_clothing * cropped_clothing[:, :, c] +
            (1 - alpha_clothing) * user_img[y_offset:y_end, x_offset:x_end, c]
        )
    return user_img

def virtual_try_on_with_pose_detection(user_image, clothing_image, output_path):
    """
    Main function for virtual try-on with pose detection and alignment.

    Parameters:
    - user_image: Path to the user's photo (str).
    - clothing_image: Path to the clothing image (str).
    - output_path: Path to save the result (str).
    """
    try:
        # Load images
        user_img = load_image(user_image)
        clothing_img = load_image(clothing_image)

        if clothing_img.shape[2] != 4:
            raise ValueError("Clothing image must have an alpha channel.")

        # Extract pose keypoints
        pose_keypoints = get_pose_keypoints(user_img)
        if len(pose_keypoints) < 13:
            raise ValueError("Insufficient pose keypoints detected.")

        # Align and resize clothing
        left_shoulder, right_shoulder = pose_keypoints[11], pose_keypoints[12]
        resized_clothing, offset = align_and_resize_clothing(clothing_img, left_shoulder, right_shoulder)

        # Overlay clothing
        result_img = overlay_clothing(user_img, resized_clothing, offset)

        # Save result
        cv2.imwrite(output_path, result_img)
        print(f"Saved output image to {output_path}")
    except Exception as e:
        print(f"Error during virtual try-on: {e}")
