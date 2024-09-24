import face_recognition
import numpy as np
import os
import cv2
import threading
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# Ruta de notificaciones
notifications_directory = "/home/eon/Documents/notifications/"
record = "record_faces.txt"
notifications_path = os.path.join(notifications_directory, record)

# Cargar caracteristicas faciales conocidas y nombres
def load_face_data(encoding_path, names_path):
    known_face_encodings = []
    known_face_names = []
    try:
        with open(encoding_path, "r") as encodings_file:
            for line in encodings_file:
                encoding = np.array([float(value) for value in line.strip().split(',')])
                known_face_encodings.append(encoding)

        with open(names_path, "r") as names_file:
            for line in names_file:
                known_face_names.append(line.strip())
    except FileNotFoundError:
        print("Archivo de datos faciales no encontrado.")
    return np.array(known_face_encodings), known_face_names
# Carga de rostros conocidos
known_face_encodings, known_face_names = load_face_data(encoding_path, names_path)
# Funcion para registrar el rostro detectado en un archivo de texto
def record_face(name, locations):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(notifications_path):
        with open(notifications_path, 'w') as file:
            file.write("Registro de rostros detectados:\n")
    with open(notifications_path, 'a') as file:
        file.write(f"Nombre: {name}, Ubicaciones: {locations}, Fecha y Hora: {current_time}\n")
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

        for face_encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Desconocido"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            # Registrar en el archivo si se detecta un rostro conocido
            if name != "Desconocido":
                record_face(name, location)
            face_names.append(name)
    
    process_this_frame = not process_this_frame
#Observador de actualizaciones para rostros
class ChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        print(f"Detected change: {event.src_path}")
        update_data_faces()

def update_data_faces():
    global known_face_encodings, known_face_names
    print("Actualizando datos faciales...")
    known_face_encodings, known_face_names = load_face_data(encoding_path, names_path)
    print("Actualizacion completada.")
# Actualiza los archivos de rostros en caso de cambios
event_handler = ChangeHandler()
observer = Observer()
observer.schedule(event_handler, path=directory, recursive=True)
observer.start()
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
    

# Liberar recursos
observer.stop()
observer.join()
cap.release()
cv2.destroyAllWindows()
