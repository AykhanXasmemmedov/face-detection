import cv2
import os
import numpy as np

names=[]
for name in os.listdir(r"C:\Users\ayxan\Pictures\face_projction\face_recogntion"):
    names.append(name)
print(names)

features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy')

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

haar_cascade=cv2.CascadeClassifier("C:/Users/ayxan/Desktop/face_detection/haarcascade_frontalface_default.xml")

cam=cv2.VideoCapture(r"C:\Users\ayxan\Pictures\face_projction\test_images\Yemin dizisinin Emir'i Gökberk Demirci'den mesajiniz var!.mp4")

while True:
    _,frame=cam.read()    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

    detected_face=haar_cascade.detectMultiScale(gray,1.1,5)
    for coordinate in detected_face:
        (x,y,w,h)=coordinate
        face=gray[y:y+h,x:x+h]
        #cv2.imshow("face",face)
        label,confidence=face_recognizer.predict(face)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,str(names[label]),(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),1)
        #cv2.putText(image,f"confidence{confidence}",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),1)

print(confidence)   
cv2.imshow("detected_frame",frame)
cv2.waitKey(0)












