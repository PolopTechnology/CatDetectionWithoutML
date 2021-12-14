import cv2
import numpy as np

img = cv2.imread("insert_file_path_here", cv2.IMREAD_UNCHANGED)
scale_percent = 40
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)

face_classifier = cv2.CascadeClassifier('cat_face.xml')
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
if faces is ():
    print("No faces found")
for (x,y,w,h) in faces:
    cv2.rectangle(resized, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', resized)
    cv2.imwrite("resultado.jpg", resized)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
