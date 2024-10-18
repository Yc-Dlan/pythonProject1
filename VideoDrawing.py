import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)
fourcc=cv.VideoWriter_fourcc('X','V','I','D')
out=cv.VideoWriter('output.avi',fourcc,30.0,(640,480))

while cap.isOpened():
    ret,frame=cap.read()
    if ret:
        frame=cv.flip(frame,1)
        out.write(frame)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()

