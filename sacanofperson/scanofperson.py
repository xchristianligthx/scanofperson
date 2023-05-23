import cv2
import numpy as np

# Crea el clasificador Haar Cascade para detectar cuerpos completos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Define una lista de colores aleatorios
colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

# Lee el video de muestra
cap = cv2.VideoCapture('sample_video.mp4')

while cap.isOpened():
    # Lee cada cuadro del video
    ret, frame = cap.read()

    if ret:
        # Convierte el cuadro a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica el clasificador al cuadro
        bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

        # Dibuja un rectángulo alrededor de cada cuerpo detectado con un color aleatorio y efecto de borde brillante
        for (x, y, w, h) in bodies:
            color = colors[np.random.choice(len(colors))]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        # Muestra el cuadro con los cuerpos detectados
        cv2.imshow('Detected Bodies', frame)

        # Si se presiona la tecla 'q', sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
