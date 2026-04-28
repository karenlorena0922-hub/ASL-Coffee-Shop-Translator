import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def draw_ui(image, estado_msg, color_borde):
    h, w = image.shape[:2]

    # Borde sutil (mas delgado y elegante)
    cv2.rectangle(image, (0, 0), (w, h), color_borde, 8)

    # Fondo semitransparente para el texto (pill shape en la parte inferior)
    overlay = image.copy()
    bar_h = 60
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Indicador de color (circulo pequeno a la izquierda del texto)
    cx, cy = 30, h - 28
    cv2.circle(image, (cx, cy), 8, color_borde, -1)

    # Texto del estado - fuente mas pequena y limpia
    cv2.putText(image, estado_msg, (52, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

    # Titulo de la app arriba - barra superior
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.6, image, 0.4, 0, image)
    cv2.putText(image, "Bidirectional Communication  |  UX Prototype",
                (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    color_borde = (80, 80, 80)
    estado_msg = "Searching..."

    if results.face_landmarks:
        if results.left_hand_landmarks and results.right_hand_landmarks:
            color_borde = (100, 220, 100)
            estado_msg = "Ready to translate"
        elif results.left_hand_landmarks or results.right_hand_landmarks:
            color_borde = (0, 200, 220)
            estado_msg = "Adjust your position"
        else:
            color_borde = (200, 160, 50)
            estado_msg = "Face detected  —  show your hands"

    draw_ui(image, estado_msg, color_borde)
    cv2.imshow('Thesis Prototype', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()