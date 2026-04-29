import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# 1. Setup paths and labels
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data'
MODEL_PATH = BASE_DIR / 'action.h5'
# Define the 10 actions used in the coffee shop scenario
actions = np.array(['hello', 'coffee', 'hot', 'ice', 'thank_you', 'nothing', 'milk', 'sugar', 'please', 'cold'])

# Dataset parameters
sequence_length = 40 # Each sample consists of 40 frames

# Create a map for labels
label_map = {label:num for num, label in enumerate(actions)}

np.random.seed(7)
tf.random.set_seed(7)

# 2. Load and preprocess data
sequences, labels = [], []
print("Loading dataset...")

for action in actions:
    sequence_dirs = sorted(
        [path for path in (DATA_PATH / action).iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name)
    )

    for sequence_dir in sequence_dirs:
        window = []
        for frame_num in range(sequence_length):
            # Load the individual frame coordinate files (.npy)
            res = np.load(sequence_dir / f"{frame_num}.npy")
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into Training (95%) and Testing (5%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=7,
    stratify=labels
)

# 3. Build the LSTM Neural Network Architecture
# This model uses LSTM layers to process the temporal sequence of the signs
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1530)))
model.add(Dropout(0.2)) # Prevents overfitting by randomly disabling neurons
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Output layer for 10 classes

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Train the model
print("Starting training process...")
# EarlyStopping stops training if the model stops improving to save time
early_stop = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=30,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001
)

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# 5. Model Summary and Save
model.summary()
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.1f}%")
model.save(MODEL_PATH)
print(f"Model trained and saved as '{MODEL_PATH}'")
