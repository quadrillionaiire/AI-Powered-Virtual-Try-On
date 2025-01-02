import json
import os

# Paths to OpenPose JSON and output directory
openpose_json_dir = "./data/raw/train/openpose_json"
mediapipe_json_dir = "./data/processed/mediapipe_json"

# Ensure the output directory exists
os.makedirs(mediapipe_json_dir, exist_ok=True)

# Mapping OpenPose (COCO) indices to MediaPipe indices
openpose_to_mediapipe = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_HIP",
    18: "RIGHT_HIP",
    19: "LEFT_KNEE",
    20: "RIGHT_KNEE",
    21: "LEFT_ANKLE",
    22: "RIGHT_ANKLE",
}

# Process JSON files
for json_file in os.listdir(openpose_json_dir):
    # Skip non-JSON files
    if not json_file.endswith((".json", ".json.html")):
        continue

    # Remove `.html` suffix if present
    clean_file_name = json_file.replace(".html", "")
    input_path = os.path.join(openpose_json_dir, json_file)
    output_path = os.path.join(mediapipe_json_dir, clean_file_name)

    with open(input_path, 'r') as f:
        try:
            openpose_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {json_file}")
            continue

    # Ensure the data contains keypoints
    if not openpose_data.get('people') or not openpose_data['people'][0].get('pose_keypoints_2d'):
        print(f"No keypoints found in {json_file}")
        continue

    keypoints = openpose_data['people'][0]['pose_keypoints_2d']
    reshaped_keypoints = [keypoints[i:i + 3] for i in range(0, len(keypoints), 3)]

    # Map to MediaPipe structure
    mediapipe_data = {}
    for i, point in enumerate(reshaped_keypoints):
        if i in openpose_to_mediapipe:
            landmark_name = openpose_to_mediapipe[i]
            mediapipe_data[landmark_name] = {
                "x": point[0],
                "y": point[1],
                "z": 0,  # OpenPose doesn't provide depth
                "visibility": point[2],
            }

    # Save as new JSON file
    with open(output_path, 'w') as f:
        json.dump(mediapipe_data, f, indent=4)

print("Conversion completed successfully.")
