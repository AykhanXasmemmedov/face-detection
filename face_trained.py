import cv2
import numpy as np
import os

Path=r'C:\Users\ayxan\Pictures\face_projction\face_recogntion\face1'
files=[]
for file in os.listdir(Path):
    files.append(file)
print(files)
HAAR_CASCADE=cv2.CascadeClassifier(r"C:\Users\ayxan\Desktop\face_detection\haarcascade_frontalface_default.xml")

features=[]
labels=[]
def create_train():
           
    for person in files:
        path=os.path.join(Path,person)
        label=files.index(person)
   
        for image in os.listdir(path):
            
            img_path=os.path.join(path,image)
            img=cv2.imread(img_path)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            
            faces_detect=HAAR_CASCADE.detectMultiScale(gray,1.1,5)
            for coordinate in faces_detect:
                (x,y,w,h)=coordinate
                face=gray[y:y+h,x:x+w]
                
               
                features.append(face)
                labels.append(label)
                
                

create_train()
#print(f"the main labels:{len(labels)}")
#print(f"the main features:{len(features)}")
features=np.array(features,dtype='object')
labels=np.array(labels)
print("trained-------------------------------------------")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml') 
np.save("features.npy",features)
np.save("labels.npy",labels)   
print("train finsihed-------------------------------------")
