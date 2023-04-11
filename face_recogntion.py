import cv2
import os
import numpy as np
from cv2 import CascadeClassifier 

names=[]
for name in os.listdir(r"C:\Users\ayxan\Pictures\face_projction\face_recogntion\face1"):
    names.append(name)
print(names)

features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy')    

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

haar_cascade=cv2.CascadeClassifier("C:/Users/ayxan/Desktop/face_detection/haarcascade_frontalface_default.xml")

image=cv2.imread(r"C:\Users\ayxan\Pictures\face_projction\test_image_SOZ\download.jfif")

#image=cv2.resize(image, (int(w/2),int(h/2)) )
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("orginal",image)
cv2.waitKey(0)
detected_face=haar_cascade.detectMultiScale(gray,1.1,4)

for coordinate in detected_face:
    (x,y,w,h)=coordinate
    face=gray[y:y+h,x:x+h]
    cv2.imshow("face",face)
    label,confidence=face_recognizer.predict(face)
    #confidence=100-float(confidence)
    if confidence>50:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(image,str(names[label]),(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),1)
    #cv2.putText(image,f"confidence{confidence}",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),1)
    else:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(image,"unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.putText(image,str(names[label]),(x,y),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),1)        

print(confidence)   
cv2.imshow("detected_image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

