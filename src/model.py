import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def build_unet(input_shape=(256, 256, 3)):
    """
    Build a U-Net model for segmentation tasks.

    Parameters:
    - input_shape: Tuple, the shape of input images.

    Returns:
    - model: A compiled U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def live_pose_detection():
    """
    Perform live pose detection using a webcam feed.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read webcam feed.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("Live Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    cap.release()
    cv2.destroyAllWindows()

def overlay_clothing(image, clothing, scaling_factor=1.0, x_offset=0, y_offset=0, use_pose_landmarks=False):
    """
    Overlay a clothing image onto a user's photo with pose-based alignment.

    Parameters:
    - image: User's photo.
    - clothing: Clothing image to overlay.
    - scaling_factor: Scale of the clothing overlay.
    - x_offset: Horizontal offset.
    - y_offset: Vertical offset.
    - use_pose_landmarks: Use pose keypoints for alignment.

    Returns:
    - image: Image with clothing overlay.
    """
    if use_pose_landmarks:
        with mp_pose.Pose(static_image_mode=True) as pose:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Use shoulder keypoints for alignment
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                shoulder_width = int(abs(right_shoulder.x - left_shoulder.x) * image.shape[1])
                x_offset = int(left_shoulder.x * image.shape[1])
                y_offset = int(left_shoulder.y * image.shape[0])
                scaling_factor = shoulder_width / clothing.shape[1]

    resized_clothing = cv2.resize(
        clothing, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA
    )

    # Overlay position
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(image.shape[1], x_start + resized_clothing.shape[1])
    y_end = min(image.shape[0], y_start + resized_clothing.shape[0])

    # Overlay clothing
    alpha_clothing = resized_clothing[:, :, 3] / 255.0  # Extract alpha channel
    for c in range(3):
        image[y_start:y_end, x_start:x_end, c] = (
            alpha_clothing * resized_clothing[:y_end - y_start, :x_end - x_start, c] +
            (1 - alpha_clothing) * image[y_start:y_end, x_start:x_end, c]
        )

    return image
