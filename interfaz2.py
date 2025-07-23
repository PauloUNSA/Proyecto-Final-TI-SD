import cv2
import threading
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox

current_frame = None
annotated_frame = None
cap = None
inference_lock = threading.Lock()
#model = YOLO("weights/last.pt")
import os, sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # carpeta temporal de PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model_path = resource_path("weights/last.pt")
model = YOLO(model_path)

target_classes = list(range(18))
class_names = model.names
stop_stream = False
# Colores RGB para cada clase (0 a 16)
class_colors = {
    0: (255, 0, 0),       # Rojo
    1: (0, 255, 0),       # Verde
    2: (0, 0, 255),       # Azul
    3: (255, 255, 0),     # Amarillo
    4: (255, 0, 255),     # Magenta
    5: (0, 255, 255),     # Cian
    6: (255, 165, 0),     # Naranja
    7: (128, 0, 128),     # Púrpura
    8: (0, 128, 128),     # Verde azulado
    9: (128, 128, 0),     # Oliva
    10: (0, 0, 128),      # Azul marino
    11: (139, 69, 19),    # Marrón
    12: (0, 100, 0),      # Verde oscuro
    13: (220, 20, 60),    # Carmesí
    14: (75, 0, 130),     # Índigo
    15: (255, 192, 203),  # Rosa
    16: (105, 105, 105),  # Gris oscuro
}

def start_inference():
    global current_frame, annotated_frame, cap, stop_stream

    def inference_loop():
        global annotated_frame, stop_stream
        while not stop_stream:
            with inference_lock:
                frame = current_frame.copy() if current_frame is not None else None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                results = model(frame, imgsz=640, conf=0.5)[0]
                temp_frame = frame.copy()
                boxes = results.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        if cls_id in target_classes:
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                            conf = boxes.conf[i].item()
                            label = f"{class_names[cls_id]} {conf:.2f}"
                            '''cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(temp_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)'''
                            color = class_colors.get(cls_id,
                                                     (0, 255, 0))  # Si por alguna razón la clase no está definida
                            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(temp_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                with inference_lock:
                    annotated_frame = temp_frame
            except Exception as e:
                print(f"⚠️ Error en inferencia: {e}")

    def display_loop():
        global current_frame, annotated_frame, stop_stream
        while not stop_stream and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            with inference_lock:
                current_frame = cv2.resize(frame, (640, 480))
                display_frame = annotated_frame if annotated_frame is not None else current_frame
            cv2.imshow("Detección en Tiempo Real", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_stream = True
                break
        cap.release()
        cv2.destroyAllWindows()

    stop_stream = False
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=display_loop, daemon=True).start()


def start_laptop_cam():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo abrir la cámara de laptop.")
        return
    start_inference()


def start_dvr_cam():
    global cap

    ip = entry_ip.get()
    user = entry_user.get()
    password = entry_pass.get()
    canal = entry_channel.get()
    port = entry_port.get()

    if not ip or not canal:
        messagebox.showerror("Error", "Debe ingresar IP y canal.")
        return

    try:
        url = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{canal}01"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el DVR. Verifique los datos.")
            return
        start_inference()
    except Exception as e:
        messagebox.showerror("Error", f"Error al abrir DVR: {e}")


# Interfaz gráfica
root = tk.Tk()
root.title("YOLOv12n - Selector de Fuente")

frame = ttk.Frame(root, padding=10)
frame.grid()

ttk.Label(frame, text="IP DVR:").grid(column=0, row=0)
entry_ip = ttk.Entry(frame, width=20)
entry_ip.grid(column=1, row=0)

ttk.Label(frame, text="Puerto:").grid(column=0, row=1)
entry_port = ttk.Entry(frame, width=10)
entry_port.insert(0, "554")
entry_port.grid(column=1, row=1)

ttk.Label(frame, text="Usuario:").grid(column=0, row=2)
entry_user = ttk.Entry(frame, width=20)
entry_user.insert(0, "admin")
entry_user.grid(column=1, row=2)

ttk.Label(frame, text="Contraseña:").grid(column=0, row=3)
entry_pass = ttk.Entry(frame, width=20, show="*")
entry_pass.insert(0, "admin")
entry_pass.grid(column=1, row=3)

ttk.Label(frame, text="Canal:").grid(column=0, row=4)
entry_channel = ttk.Entry(frame, width=5)
entry_channel.insert(0, "1")
entry_channel.grid(column=1, row=4)

ttk.Button(frame, text="Usar Cámara de Laptop", command=start_laptop_cam).grid(column=0, row=5, columnspan=2, pady=10)
ttk.Button(frame, text="Conectar al DVR", command=start_dvr_cam).grid(column=0, row=6, columnspan=2)

root.mainloop()
