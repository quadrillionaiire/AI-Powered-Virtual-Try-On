# Pose Detection Code
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Save annotated image
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to {output_path}")
        else:
            print("No pose detected.")

