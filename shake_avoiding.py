import cv2 as cv
import numpy as np

cap = cv.VideoCapture('image/shaketest2.mp4')

ret, old_frame = cap.read()
if not ret:
    print("Cannot read video file or stream.")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

cv.namedWindow('Warped Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Warped Frame', 1920, 1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if old_gray.shape != frame_gray.shape:
        print("Error: The size of the two frames is different.")
        break

    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.5, cv.OPTFLOW_FARNEBACK_GAUSSIAN)

    points = np.column_stack(np.where(np.abs(flow[..., 0]) > 0.5))
    if points.shape[1] >= 4:
        M, inliers = cv.estimateAffinePartial2D(points, points + flow[points[:, 0].astype(int), points[:, 1].astype(int)])

        if M is not None:
            translation = M[:, 2]
            translation_gaussian = cv.GaussianBlur(translation.reshape(-1, 1), (5, 5), 0)
            translation_gaussian = translation_gaussian.squeeze()
            M[:, 2] = translation_gaussian
            warped_img = cv.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        else:
            warped_img = frame
    else:
        warped_img = frame

    cv.imshow('Warped Frame', warped_img)

    if cv.waitKey(1) == ord('q'):
        break

    old_gray = frame_gray.copy()

cap.release()
cv.destroyAllWindows()