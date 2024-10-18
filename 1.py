import cv2 as cv
import numpy as np

cap = cv.VideoCapture('image/fangdou.mp4')
orb = cv.ORB_create()
fps = cap.get(cv.CAP_PROP_FPS)
ret, old_frame = cap.read()
if not ret:
    print("Cannot read video file or stream.")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
key1 = orb.detect(old_gray)
key1, des1 = orb.compute(old_gray, key1)
p0 = np.float32([keypoint.pt for keypoint in key1]).reshape(-1, 1, 2)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cv.namedWindow('Warped Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Warped Frame', 1920, 1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    key=orb.detect(frame_gray,None)
    key, des=orb.compute(frame_gray,key)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    M, inliers = cv.estimateAffinePartial2D(good_old, good_new)

    if M is not None:
        translation = M[:, 2]
        translation_gaussian = cv.GaussianBlur(translation.reshape(-1, 1), (5, 5), 0)
        translation_gaussian = translation_gaussian.squeeze()
        M[:, 2] = translation_gaussian
        warped_img = cv.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        cv.imshow('Warped Frame', warped_img)

    if cv.waitKey(15) == ord('q'):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()