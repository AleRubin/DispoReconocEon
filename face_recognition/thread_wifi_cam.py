import face_recognition
import numpy as np
import os
import cv2
import threading


#92.168.1.92 dir cam1,  psw eonboxseg1
#192.168.1.115 cam2 ... psw eonboxseg1
#192.168.1.81 cam3 ... psw Eonboxseg1
camera_url = "rtsp://admin:eonboxseg1@192.168.1.92/H264?ch=1&subtype=0"
cap = cv2.VideoCapture(camera_url)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("Error: No se puede abrir el flujo de la camara")
    exit(1)

# Rutas de archivos con caracteristicas faciales
directory = "/home/eon/Documents/data-ras/Face_encodings/data_faces/"
encoding_file = "known_face_encodings.txt"
names_file = "known_face_names.txt"
encoding_path = os.path.join(directory, encoding_file)
names_path = os.path.join(directory, names_file)

# Cargar caracteristicas faciales conocidas y nombres
def load_face_data(encoding_path, names_path):
    known_face_encodings = []
    known_face_names = []

    with open(encoding_path, "r") as encodings_file:
        for line in encodings_file:
            encoding = np.array([float(value) for value in line.strip().split(',')])
            known_face_encodings.append(encoding)

    with open(names_path, "r") as names_file:
        for line in names_file:
            known_face_names.append(line.strip())

    return np.array(known_face_encodings), known_face_names

known_face_encodings, known_face_names = load_face_data(encoding_path, names_path)

# Variables compartidas para el hilo de procesamiento
face_locations = []
face_names = []
process_this_frame = True

# Funcion para procesar cada frame
def process_frame(frame):
    global face_locations, face_names, process_this_frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) != 0:
            print(f"Encontrados {len(face_locations)} rostros.")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Desconocido"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    
    process_this_frame = not process_this_frame

# Iniciar la camara
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir el fotograma.")
        break

    # Redimensionar el fotograma a 640x480
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Crear un hilo para el procesamiento
    processing_thread = threading.Thread(target=process_frame, args=(frame_resized,))
    processing_thread.start()
    processing_thread.join()
    
    # Mostrar los resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
        # Dibujar una etiqueta con el nombre debajo de la cara
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
