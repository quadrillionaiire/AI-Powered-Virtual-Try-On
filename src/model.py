import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_image_pairs(data, image_folder, cloth_folder, processed_folder, img_size):
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing image pairs"):
        img_path = os.path.join(image_folder, row['image'])
        cloth_path = os.path.join(cloth_folder, row['cloth'])

        processed_img_path = os.path.join(processed_folder, "image", row['image'])
        processed_cloth_path = os.path.join(processed_folder, "cloth", row['cloth'])

        preprocess_image(img_path, processed_img_path, img_size)
        preprocess_image(cloth_path, processed_cloth_path, img_size)

def run_mediapipe_on_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    annotated_image = img.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()