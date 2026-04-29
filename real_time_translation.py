import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tensorflow.keras.models import load_model

# 1. Safe paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "action.h5"
CAMERA_INDEXES = (0, 1, 2)

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

def swap_hands(sequence_data):
    """Return a copy with left-hand and right-hand landmarks exchanged."""
    swapped = np.array(sequence_data, copy=True)
    left_start = 468 * 3
    left_end = left_start + (21 * 3)
    right_start = left_end
    right_end = right_start + (21 * 3)

    swapped[:, left_start:left_end] = sequence_data[:, right_start:right_end]
    swapped[:, right_start:right_end] = sequence_data[:, left_start:left_end]
    return swapped

def predict_with_hand_swap(sequence_data):
    """Try the original hand side and the opposite hand side, then keep the stronger prediction."""
    original = np.array(sequence_data)
    swapped = swap_hands(original)
    batch = np.array([original, swapped])
    predictions = model.predict(batch, verbose=0)

    original_confidence = np.max(predictions[0])
    swapped_confidence = np.max(predictions[1])

    if swapped_confidence > original_confidence:
        return predictions[1], "swapped"
    return predictions[0], "normal"

def format_top_predictions(prediction, count=3):
    top_indexes = np.argsort(prediction)[-count:][::-1]
    return " | ".join(
        f"{actions[index]} {int(prediction[index] * 100)}%"
        for index in top_indexes
    )

# 5. Real-time Logic Variables
sequence = []
threshold = 0.6
sequence_length = 40
use_hand_swap = False
last_prediction = "Waiting for model..."

def open_camera(indexes):
    """Open the first available camera from a small list of common indexes."""
    for index in indexes:
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            print(f"Using camera index: {index}")
            return camera
        camera.release()
    return None

cap = open_camera(CAMERA_INDEXES)

if cap is None:
    raise RuntimeError(f"Camera not found. Tried indexes: {CAMERA_INDEXES}")

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

    # UI default status
    face_seen = results.face_landmarks is not None
    left_seen = results.left_hand_landmarks is not None
    right_seen = results.right_hand_landmarks is not None
    hand_seen = left_seen or right_seen
    status_msg = f"Buffer: {len(sequence)}/{sequence_length} frames"
    detection_msg = f"Face:{'Y' if face_seen else 'N'}  Left:{'Y' if left_seen else 'N'}  Right:{'Y' if right_seen else 'N'}"
    border_color = (150, 150, 150)

    if hand_seen:
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        status_msg = f"Buffer: {len(sequence)}/{sequence_length} frames"
    else:
        sequence.clear()
        status_msg = "Show your hand inside the camera frame"
        last_prediction = "No hand detected"
        border_color = (0, 200, 255)

    # Prediction Logic
    if len(sequence) == sequence_length:
        if use_hand_swap:
            res, hand_mode = predict_with_hand_swap(sequence)
        else:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            hand_mode = "normal"

        current_idx = np.argmax(res)
        confidence = res[current_idx]
        action_name = actions[current_idx]
        last_prediction = f"Top: {format_top_predictions(res)} ({hand_mode})"

        if confidence > threshold:
            if action_name != "nothing":
                status_msg = f"ORDER: {action_name.upper()} ({int(confidence * 100)}%)"
                border_color = (0, 255, 0)
            else:
                status_msg = "IDLE - Waiting for your order"
                border_color = (255, 255, 255)

    # Visual Interface
    h, w = image.shape[:2]

    cv2.rectangle(image, (0, 0), (w, h), border_color, 12)
    cv2.rectangle(image, (0, 0), (w, 64), (30, 30, 30), -1)
    cv2.putText(
        image,
        detection_msg,
        (20, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        1
    )
    cv2.putText(
        image,
        last_prediction,
        (20, 52),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        1
    )
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
