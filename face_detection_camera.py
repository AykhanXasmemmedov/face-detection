import numpy as np
import cv2
def nothing(x):
    pass
cv2.namedWindow("Frame")
cv2.createTrackbar("Scale","Frame",11,20,nothing)
cv2.createTrackbar("neighbors","Frame",0,20,nothing)
haarcadcade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera=cv2.VideoCapture(0)
while True:
    _,frame=camera.read()
    #frame=cv2.flip(frame,1)
    cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scale=cv2.getTrackbarPos('Scale','Frame')
    neighors=cv2.getTrackbarPos('neighbors','Frame')
    if scale/10>1.0:
        faces=haarcadcade.detectMultiScale(frame,scale/10,neighors)
    print(faces)
    print(f"faces number:{len(faces)}")
    for coordinate in faces:
        (x,y,w,h)=coordinate
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)
        color_img=frame[y:y+h,x:x+w]
        color_item="image.png"
        cv2.imwrite(color_item,color_img)
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
