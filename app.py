from flask import Flask
from flask_socketio import SocketIO, emit
import base64, time, threading, cv2, os
from ultralytics import YOLO
import datetime
from flask import request, jsonify

import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CAMERA_ID = 0
processing = False
model = YOLO("model/best.pt")
stop_signal = False
video_processing = False
video_paused = False
video_thread = None
video_frames_buffer = []
last_video_path = None  # <-- thêm global để lưu video trước đó


def color_box(class_name):
    colors = {
        "gloves": (0, 255, 0),
        "hands": (255, 128, 0),
        "head": (128, 0, 128),
        "helmet": (0, 0, 255),
        "no-gloves": (255, 0, 0),
        "no-helmet": (255, 0, 255),
        "no-safety-vest": (255, 255, 0),
        "person": (0, 255, 255),
        "safety-suit": (128, 255, 0),
        "safety-vest": (0, 128, 255)
    }
    return colors.get(class_name, (255, 255, 255))  # Mặc định là trắng nếu không khớp

@socketio.on("start_camera")
def handle_start_camera(data):
    global processing, stop_signal, CAMERA_ID
    CAMERA_ID = data.get("camera_index", 0)
    if not processing:
        stop_signal = False
        print(f"Client requested to start camera #{CAMERA_ID}")
        threading.Thread(target=process_stream).start()

@socketio.on("stop_camera")
def handle_stop_camera():
    global stop_signal
    print("Client requested to stop camera")
    stop_signal = True

@socketio.on("pause_video")
def pause_video():
    global video_paused
    video_paused = True
    print("[SERVER] Video detection paused.")

@socketio.on("continue_video")
def continue_video():
    global video_paused
    video_paused = False
    print("[SERVER] Video detection continued.")

@socketio.on("clear_video")
def clear_video():
    global video_processing, video_paused
    video_processing = False
    video_paused = False
    video_frames_buffer.clear()
    print("[SERVER] Video detection cleared.")

def process_stream():
    global processing, stop_signal
    processing = True
    cap = cv2.VideoCapture(CAMERA_ID)

    while not stop_signal and cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        inference_time = round(results.speed["inference"], 2)
        detections = []
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            conf = round(float(box.conf) * 100, 1)
            xyxy = box.xyxy[0].tolist()
            detections.append({"class": cls, "confidence": conf, "box": xyxy})

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            color = color_box(det["class"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f'{det["class"]} {det["confidence"]}%'

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.resize(frame, (854, 450))

        _, buffer = cv2.imencode('.jpg', frame)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        socketio.emit("result_realtime", {
            "image": encoded_img,
            "detections": detections,
            "avg_inference_time": inference_time,
        })

        socketio.sleep(0.4)

    cap.release()
    processing = False
    print("Camera stream stopped")

    socketio.emit("clear_frame_realtime")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    print(f"[UPLOAD] Video saved to: {save_path}")

    # Gọi trực tiếp
    threading.Thread(target=process_video, args=(save_path,)).start()

    return jsonify({"success": True, "path": save_path})

def process_video(path):
    global video_processing, video_paused, video_frames_buffer
    if not os.path.exists(path) or not os.path.isfile(path):
        print("[ERROR] Invalid file path")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    video_processing = True
    video_paused = False
    video_frames_buffer.clear()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames: {total_frames}")

    inferences_time = []
    while video_processing and cap.isOpened():
        if video_paused:
            print("[INFO] Video detection paused. Waiting to continue...")
            socketio.sleep(1)
            continue

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame >= total_frames:
            print("[INFO] Reached end of video")
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame read failed")
            break

        results = model(frame)[0]
        inference_time = round(results.speed["inference"], 2)
        inferences_time.append(inference_time)
        avg_inference_time = round(sum(inferences_time) / len(inferences_time), 2)
        detections = []
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            conf = round(float(box.conf) * 100, 1)
            xyxy = box.xyxy[0].tolist()
            detections.append({"class": cls, "confidence": conf, "box": xyxy})

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            color = color_box(det["class"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{det["class"]} {det["confidence"]}%'
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.resize(frame, (854, 450))
        video_frames_buffer.append(frame.copy())

        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        socketio.emit("result_video", {
            "image": base64_frame,
            "detections": detections,
            "avg_inference_time": avg_inference_time,
        })

        print(f"[INFO] Processed frame {current_frame + 1}/{total_frames}")
        socketio.sleep(0.2)

    cap.release()
    video_processing = False
    socketio.emit("video_done")
    print("[✅] Video detection complete and sent to client.")

@socketio.on("render_video")
def handle_render_video():
    if not video_frames_buffer:
        print("[WARN] No frames available to render.")
        return

    # Tạo thư mục nếu chưa có
    output_dir = "renders"
    os.makedirs(output_dir, exist_ok=True)

    # Tạo tên file theo timestamp
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    full_path = os.path.join(output_dir, filename)

    # Lấy kích thước frame
    height, width, _ = video_frames_buffer[0].shape
    out = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for frame in video_frames_buffer:
        out.write(frame)
    out.release()

    print(f"[✅] Rendered video saved to: {full_path}")

    with open(full_path, "rb") as f:
        encoded_video = base64.b64encode(f.read()).decode('utf-8')

    socketio.emit("render_done", {
        "filename": filename,
        "video_base64": encoded_video
    })

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)