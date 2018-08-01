import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
face_num=1
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    j=0
    faces_roi = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow('frames',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
        cv2.imwrite('face'+str(face_num)+'.jpg', crop_img)
        face_num+=1
        continue
cap.release()
cv2.destroyAllWindows() 
