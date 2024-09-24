import face_recognition
from picamera2 import Picamera2
#from picamera.array import PiRGBArray
import numpy as np
import os
import cv2
# Configuracion de la camara
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
#output = PiRGBArray(camera, size=(640, 480))
#output = np.empty((640, 280, 3), dtype=np.uint8)
# Archivos con caracteristicas faciales
directory = "/home/eon/Documents/data-ras/Face_encodings/"
encoding_file="known_face_encodings.txt"
names_file="known_face_names.txt"
encoding_path = os.path.join(directory, encoding_file)
names_path = os.path.join(directory, names_file)
# Listas para obtener las caracteristicas
known_face_encodings = []
known_face_names = []
# Leer las codificaciones faciales desde el archivo txt
with open(encoding_path, "r") as encodings_file:
    for line in encodings_file:
        encoding = [float(value) for value in line.strip().split(',')]
        known_face_encodings.append(encoding)
# Leer los nombres desde el archivo txt
with open(names_path, "r") as names_file:
    for line in names_file:
        known_face_names.append(line.strip())
# Bandera para leer cada dos frames
process_this_frame = True 
# Iniciar la camara
camera.start()       
while True:
    print("Capturing image.")
    #camera.capture(output, format="bgr")
    image = camera.capture_array("main")
    if process_this_frame:
        face_locations = face_recognition.face_locations(image)
        print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # Ver si la cara es una coincidencia para la(s) cara(s) conocida(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"
            # O en lugar de eso, usar la cara conocida con la menor distancia a la nueva cara
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame 
# Mostrar los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Escalar las ubicaciones de las caras de nuevo al tamano original ya que el cuadro detectado estaba escalado a 1/4 del tamano
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        # Dibujar una etiqueta con un nombre debajo de la cara
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Mostrar la imagen resultante
    cv2.imshow('Video', image)    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
# Liberar recursos
cv2.destroyAllWindows()
camera.stop()
