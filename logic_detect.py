from ultralytics import YOLO

model = YOLO("model/PPE_Violence_Detection_Test_2.pt")

def detect_violence(frame):
    results = model.predict(source=frame, show=True, conf=0.6)
    return results[0].plot()