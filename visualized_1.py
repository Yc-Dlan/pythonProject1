import cv2 as cv
import numpy as np

cap = cv.VideoCapture('image/fangdou.mp4')

orb=cv.ORB_create()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv.imshow('origin',frame)
        frame_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        kp=orb.detect(frame_gray,None)
        kp, des=orb.compute(frame_gray,kp)

        visual=cv.drawKeypoints(frame_gray,kp,frame_gray,(0,255,0),flags=0)
        cv.imshow('visualized',visual)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:kodjiahyaif
        break

cap.release()
cv.destroyAllWindows()