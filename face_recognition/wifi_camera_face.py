import face_recognition
import numpy as np
import os
import cv2
# camara Exterior
cap = cv2.VideoCapture()
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Ajustar el tamano del buffer si es necesario
#92.168.1.92 dir cam1,  psw eonboxseg1
#192.168.1.115 cam2 ... psw eonboxseg1
#192.168.1.81 cam3 ... psw Eonboxseg1
#
cap.open('rtsp://admin:eonboxseg1@192.168.1.92/H264?ch=1&subtype=0')
if not cap.isOpened():
    print("Error: No se puede abrir el flujo de la camara")
    exit(1)
# Archivos con caracteristicas faciales
directory = "/home/eon/Documents/data-ras/Face_encodings/"
encoding_file="known_face_encodings.txt"
names_file="known_face_names.txt"
encoding_path = os.path.join(directory, encoding_file)
names_path = os.path.join(directory, names_file)
def load_face_data(encoding_path, names_path):
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
    return known_face_encodings, known_face_names
known_face_encodings, known_face_names = load_face_data(encoding_path, names_path)
# Bandera para leer cada dos frames
process_this_frame = True 
# Iniciar la camara
while True:
    ret, frame = cap.read()
    #if not ret:
    #    print("Error: No se puede recibir el fotograma.")
    #    break
    # Redimensionar el fotograma a 640x480
    frame_resized = cv2.resize(frame, (640, 480))
    if process_this_frame:
        face_locations = face_recognition.face_locations(frame_resized)
        if len(face_locations) != 0:
            print(f"Encontrados {len(face_locations)} rostros.")
        face_encodings = face_recognition.face_encodings(frame_resized, face_locations)
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
        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
        # Dibujar una etiqueta con un nombre debajo de la cara
        cv2.rectangle(frame_resized, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_resized, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Mostrar el video
    cv2.imshow('Video', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar recursos
cap.release()
cv2.destroyAllWindows()       
