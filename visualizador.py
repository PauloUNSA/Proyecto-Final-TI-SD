import cv2
from ultralytics import YOLO
import time
import threading

current_frame = None
annotated_frame = None
inference_lock = threading.Lock()
'''
esto es para el DVR, esto lo probare yo, se debe de cambiar los valores, con los tomados en la interfaz 
formato "rstp://<usuario>:<contra>@<IP>:<puerto>/Streaming/Channels/<canal>01" 
'''
#url = "rtsp://admin:@dmin123@192.168.0.4:554/Streaming/Channels/101"
#cap = cv2.VideoCapture(url)
#esto es para la camara integrada en la laptop
cap = cv2.VideoCapture(0)

model = YOLO("weights/last.pt")
target_classes = [0, 1,2,3,4,5,6,7,8,9,10]
class_names = model.names

# Hilo de inferencia
def inference_thread():
    global current_frame, annotated_frame

    while True:
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

                        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(temp_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            with inference_lock:
                annotated_frame = temp_frame

        except Exception as e:
            print(f"⚠️ Error en hilo de inferencia: {e}")

# Iniciar hilo de inferencia
threading.Thread(target=inference_thread, daemon=True).start()


# Hilo principal: lectura y visualización
if not cap.isOpened():
    print("❌ No se pudo conectar al RTSP.")
    exit()

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo capturar el video.")
        continue

    with inference_lock:
        current_frame = cv2.resize(frame, (640, 480))

    with inference_lock:
        display_frame = annotated_frame if annotated_frame is not None else current_frame

    cv2.imshow("YOLOv12n RTSP Inferencia Asíncrona", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()