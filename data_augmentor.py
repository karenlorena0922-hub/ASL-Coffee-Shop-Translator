import numpy as np
import os

# --- CONFIGURATION ---
DATA_PATH = 'D:/Tesis/data'
actions = ['hello', 'coffee','hot', 'ice', 'thank_you', 'nothing', 'milk', 'sugar', 'please','cold',]
no_sequences = 50 # Tus muestras originales
augmentation_factor = 2 # Cuántas variaciones nuevas crear por cada una
sequence_length = 40

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def shift_data(data, shift_range=0.05):
    shift = np.random.uniform(-shift_range, shift_range, (1, 3)) # Shift en X, Y, Z
    # Necesitamos reformatear para aplicar el shift solo a las coordenadas
    reshaped = data.reshape(-1, 3)
    reshaped += shift
    return reshaped.flatten()

print("Starting Data Augmentation...")

for action in actions:
    print(f"Augmenting action: {action}")
    for sequence in range(no_sequences):
        source_dir = os.path.join(DATA_PATH, action, str(sequence))
        
        for i in range(augmentation_factor):
            # Crear nueva carpeta para la muestra aumentada (ej: 30, 31, 32...)
            new_sequence_num = no_sequences + (sequence * augmentation_factor) + i
            new_dir = os.path.join(DATA_PATH, action, str(new_sequence_num))
            os.makedirs(new_dir, exist_ok=True)
            
            for frame_num in range(sequence_length):
                # Cargar el frame original
                frame_path = os.path.join(source_dir, f"{frame_num}.npy")
                data = np.load(frame_path)
                
                # Aplicar transformaciones
                augmented_data = shift_data(data)
                augmented_data = add_noise(augmented_data)
                
                # Guardar en la nueva carpeta
                np.save(os.path.join(new_dir, f"{frame_num}.npy"), augmented_data)

print("Finished! Your dataset is now 3x larger.")