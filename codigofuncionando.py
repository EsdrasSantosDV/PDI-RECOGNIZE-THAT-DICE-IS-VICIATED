import cv2
import numpy as np
from collections import deque
min_threshold = 10
max_threshold = 200
min_area = 60
min_circularity = 0.3
min_inertia_ratio = 0.5
cap =cv2.VideoCapture('video9.mp4')
cap.set(15, -4)
i=1
counter = 0
readings = deque([0, 0], maxlen=10)
display = deque([0, 0], maxlen=10)
while True:
    ret, im = cap.read()
    if ret == True:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.minArea = min_area
        params.minCircularity = min_circularity
        params.minInertiaRatio = min_inertia_ratio
        params.filterByColor = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(im)
        if len(keypoints):
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 360, 640)
            cv2.imshow("frame", im_with_keypoints)


        if counter % 10 == 0:
            reading = len(keypoints)
            readings.append(reading)

            if readings[-1] == readings[-2] == readings[-3]:
                display.append(readings[-1])


            if display[-1] != display[-2] and display[-1] != 0:
                msg = f"jogada {i} --Resultado=> {display[-1]}\n****"
                i = i + 1
                print(msg)

        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()