import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "action.h5"

actions = np.array([
    "hello", "coffee", "hot", "ice", "thank_you",
    "nothing", "milk", "sugar", "please", "cold",
])

sequence_length = 40


def load_sequence(sequence_dir):
    frames = []
    for frame_num in range(sequence_length):
        frames.append(np.load(sequence_dir / f"{frame_num}.npy"))
    return np.array(frames)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    confusion = np.zeros((len(actions), len(actions)), dtype=int)

    print(f"Testing model: {MODEL_PATH}")
    print(f"Testing data:  {DATA_PATH}")
    print()

    for actual_index, action in enumerate(actions):
        correct = 0
        confidences = []
        sequence_dirs = sorted(
            [path for path in (DATA_PATH / action).iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name)
        )

        for sequence_dir in sequence_dirs:
            sample = load_sequence(sequence_dir)
            prediction = model.predict(np.expand_dims(sample, axis=0), verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            confidence = float(prediction[predicted_index])

            confusion[actual_index, predicted_index] += 1
            confidences.append(confidence)

            if predicted_index == actual_index:
                correct += 1

        accuracy = correct / len(sequence_dirs)
        avg_confidence = sum(confidences) / len(confidences)
        print(f"{action:10s} accuracy: {accuracy * 100:5.1f}% | avg confidence: {avg_confidence * 100:5.1f}%")

    total_correct = int(np.trace(confusion))
    total = int(confusion.sum())
    print()
    print(f"Overall accuracy: {total_correct / total * 100:.1f}% ({total_correct}/{total})")
    print()
    print("Confusion matrix")
    print("rows = real sign, columns = predicted sign")
    print(" " * 12 + " ".join(f"{action[:5]:>5s}" for action in actions))
    for row_index, action in enumerate(actions):
        counts = " ".join(f"{value:5d}" for value in confusion[row_index])
        print(f"{action[:10]:>10s}  {counts}")


if __name__ == "__main__":
    main()
