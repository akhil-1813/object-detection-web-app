import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

cap = None
camera_running = False
latest_frame = None
object_count = 0


def generate_frames():
    global cap, latest_frame, object_count, camera_running

    while True:
        if not camera_running or cap is None:
            continue

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        # Run detection
        results = model(frame, imgsz=640, conf=0.6)

        annotated_frame = frame.copy()
        object_count = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            # Ignore wrong class
            if label == "toothbrush":
                continue

            object_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(annotated_frame,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        latest_frame = annotated_frame.copy()

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global cap, camera_running

    if not camera_running:
        cap = cv2.VideoCapture(0)
        camera_running = True

    return "Started"


@app.route('/stop')
def stop():
    global cap, camera_running

    camera_running = False

    if cap:
        cap.release()
        cap = None

    return "Stopped"


@app.route('/count')
def count():
    return str(object_count)


@app.route('/capture')
def capture():
    global latest_frame

    if latest_frame is not None:
        cv2.imwrite("capture.jpg", latest_frame)
        return "Captured"

    return "No Frame"


if __name__ == "__main__":
    app.run(debug=True)