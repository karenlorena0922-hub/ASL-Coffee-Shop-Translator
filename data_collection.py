import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

# 1. Configuración de Acciones
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data'
actions = ['hello', 'coffee','hot', 'ice', 'thank_you', 'nothing', 'milk', 'sugar', 'please','cold',]
sequence_length = 40 # 1 segundo por muestra
current_action_idx = 0

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(2)

def next_sample_number(action):
    action_path = DATA_PATH / action
    action_path.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.name)
        for path in action_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return max(existing_numbers, default=-1) + 1

print("INSTRUCCIONES:")
print(" - Presiona 'N' para cambiar a la siguiente palabra")
print(" - MANTÉN PRESIONADA la tecla 'R' para grabar una muestra (30 frames)")
print(" - Presiona 'Q' para salir")

while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    action = actions[current_action_idx]
    sample_counter = next_sample_number(action)
    
    # UI informativa
    cv2.putText(image, f'ACCION: {action.upper()}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(image, f'Proxima muestra: {sample_counter}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('Karen - Grabador Manual para Tesis', image)
    
    key = cv2.waitKey(10) & 0xFF
    
    # Cambiar de palabra
    if key == ord('n'):
        current_action_idx = (current_action_idx + 1) % len(actions)
        print(f"Cambiado a: {actions[current_action_idx]}")

    # GRABAR (solo si presionas R)
    if key == ord('r'):
        print(f"Grabando muestra {sample_counter} para {action}...")
        path = DATA_PATH / action / str(sample_counter)
        os.makedirs(path, exist_ok=True)
        
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            # Lógica de extracción (igual a la anterior)
            image_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(image_proc)
            
            face = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark]).flatten() if res.face_landmarks else np.zeros(468*3)
            lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark]).flatten() if res.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark]).flatten() if res.right_hand_landmarks else np.zeros(21*3)
            
            keypoints = np.concatenate([face, lh, rh])
            np.save(path / str(frame_num), keypoints)
            
            # Feedback visual rápido
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.imshow('Karen - Grabador Manual para Tesis', frame)
            cv2.waitKey(1)
            
        sample_counter += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
