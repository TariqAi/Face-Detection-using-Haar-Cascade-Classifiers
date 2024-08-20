import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]       
        roi_color = frame[y:y+h, x:x+w]       

    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
