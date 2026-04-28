import cv2
import mediapipe as mp

# Usamos la forma más directa de llamar a las herramientas
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configuramos el "cerebro" (Holistic detecta cara, manos y cuerpo)
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

print("Iniciando cámara... mira a la lente.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Procesar la imagen
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- MÉTRICAS DE UX PARA TU TESIS ---
    # 1. Feedback Visual (Borde de color)
    # Si detecta rostro (NMS - Non Manuals), borde verde. Si no, rojo.
    color_borde = (0, 255, 0) if results.face_landmarks else (0, 0, 255)
    cv2.rectangle(image, (0,0), (image.shape[1], image.shape[0]), color_borde, 20)

    # Mostrar la ventana
    cv2.imshow('Tesis UX - Sistema de Comunicacion', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()