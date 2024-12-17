import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Live Pose Detection with OpenCV to analyze a live camera feed

def live_pose_detection():
    cap = cv2.VideoCapture(0)  # Open the webcam
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read the webcam feed.")
                break

            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    cap.release()
    cv2.destroyAllWindows()

# Overlay Clothing Images and map clothing images based on detected pose keypoints 
def overlay_clothing(image, clothing, landmarks, x_offset=0, y_offset=0):
    """
    Align clothing on body keypoints detected from pose landmarks.
    """
    # Extract landmarks for shoulders (example points: 11, 12)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Calculate alignment based on shoulder positions
    x1, y1 = int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])
    x2, y2 = int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])

    # Resize clothing image to fit between shoulders
    width = abs(x2 - x1) + 100
    height = int(width * (clothing.shape[0] / clothing.shape[1]))
    resized_clothing = cv2.resize(clothing, (width, height))

    # Overlay clothing image
    x_start = x1 + x_offset
    y_start = y1 + y_offset
    image[y_start:y_start+height, x_start:x_start+width] = resized_clothing

    return image
