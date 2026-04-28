import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tensorflow.keras.models import load_model

# 1. Safe paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "action.h5"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# 2. Configuration and Model Loading
actions = np.array([
    'hello', 'coffee', 'hot', 'ice', 'thank_you',
    'nothing', 'milk', 'sugar', 'please', 'cold'
])

model = load_model(MODEL_PATH)

# 3. Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 4. Keypoint Extraction Function
def extract_keypoints(results):
    """Converts MediaPipe results into a flat array for the AI model."""
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([face, lh, rh])

# 5. Real-time Logic Variables
sequence = []
threshold = 0.6
sequence_length = 40

# Try camera 2 first. If it does not work, change this to 0 or 1.
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError("Camera not found. Try changing cv2.VideoCapture(2) to 0 or 1.")

print("--- Karen's Coffee Shop: Translator Active ---")
print(f"Using model: {MODEL_PATH}")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Frame not received from camera.")
        continue

    # Pre-process image for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract coordinates and add to buffer
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    # UI default status
    status_msg = f"Buffer: {len(sequence)}/{sequence_length} frames"
    border_color = (150, 150, 150)

    # Prediction Logic
    if len(sequence) == sequence_length:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        current_idx = np.argmax(res)
        confidence = res[current_idx]

        if confidence > threshold:
            action_name = actions[current_idx]

            if action_name != "nothing":
                status_msg = f"ORDER: {action_name.upper()} ({int(confidence * 100)}%)"
                border_color = (0, 255, 0)
            else:
                status_msg = "IDLE - Waiting for your order"
                border_color = (255, 255, 255)

    # Visual Interface
    h, w = image.shape[:2]

    cv2.rectangle(image, (0, 0), (w, h), border_color, 12)
    cv2.rectangle(image, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(
        image,
        status_msg,
        (20, h - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        1
    )

    cv2.imshow("Karen Coffee Shop - Real Time Translator", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()