import cv2
from ultralytics import YOLO

# Loads a pre trained version of the yolo v8
model = YOLO("yolov8n.pt")  # Modelo pequeno e rápido
# Sets the video capturing device to video0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Reads the frame and predicts a list of objects in the frame (bounding boxes, classes and their probas)
    ret, frame = cap.read()

    if not ret:
        break
    
    # Predicts a list of objects containing in the frame
    results = model(frame)
    
    # Gets all bounding boxes and trustness to draw it ontop of the frame
    for result in results:
        for box in result.boxes:
            # Bounding boxes coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            # Trustness
            conf = box.conf[0].item() 
            # Gets the label
            cls = int(box.cls[0])
            # Only shows the objects within a 0.6 treshold to avoid errors
            if conf >= 0.6:
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Shows the result
    cv2.imshow("DEMO", frame)
    
    # Quitting the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
