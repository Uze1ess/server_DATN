import cv2

ip = "192.168.88.1"
user = "admin"
code = "ABCDEF"  # mã xác minh trên nhãn
url = f"rtsp://{user}:{code}@{ip}:554/ch1/main"

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Không thể kết nối RTSP")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("CS-C6N Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
