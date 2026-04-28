import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Setup paths and labels
DATA_PATH = os.path.join('D:/Tesis/data') 
# Define the 10 actions used in the coffee shop scenario
actions = np.array(['hello', 'coffee', 'hot', 'ice', 'thank_you', 'nothing', 'milk', 'sugar', 'please', 'cold'])

# Dataset parameters
no_sequences = 150  # 50 original samples + 100 augmented samples
sequence_length = 40 # Each sample consists of 40 frames

# Create a map for labels
label_map = {label:num for num, label in enumerate(actions)}

# 2. Load and preprocess data
sequences, labels = [], []
print("Loading dataset...")

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the individual frame coordinate files (.npy)
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into Training (95%) and Testing (5%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. Build the LSTM Neural Network Architecture
# This model uses LSTM layers to process the temporal sequence of the signs
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40, 1530)))
model.add(Dropout(0.2)) # Prevents overfitting by randomly disabling neurons
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Output layer for 10 classes

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Train the model
print("Starting training process...")
# EarlyStopping stops training if the model stops improving to save time
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=20)

model.fit(X_train, y_train, epochs=200, callbacks=[early_stop])

# 5. Model Summary and Save
model.summary()
model.save('action.h5')
print("Model trained and saved as 'action.h5'")