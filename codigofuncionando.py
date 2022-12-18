import cv2
import numpy as np
from collections import deque
from scipy.stats import chi2_contingency, chisquare

min_threshold = 10
max_threshold = 200
min_area = 60
max_area = 135
min_circularity = 0.2
min_inertia_ratio = 0.55
cap =cv2.VideoCapture('36jogadas.mp4')
cap.set(15, -4)
i=1

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
counter = 0
readings = deque([0, 0, 0, 0, 0, 0, 0, 0, 0,], maxlen=10)
display = deque([0, 0], maxlen=10)


leituras={
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}


while True:
    ret, im = cap.read()
    if not ret:
        break
    if ret == True:

        keypoints = detector.detect(im)
        if len(keypoints):
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 360, 640)
            cv2.imshow("frame", im_with_keypoints)
        if counter % 2== 0:
            reading = len(keypoints)
            readings.append(reading)

            if readings[-1] == readings[-2] == readings[-3] == readings[-4]== readings[-5]== readings[-6]== readings[-7]== readings[-8]== readings[-9]:
                display.append(readings[-1])


            if display[-1] != display[-2] and display[-1] != 0:
                msg = f"jogada {i} --Resultado=> LADO {display[-1]}\n"
                leituras[display[-1]] = leituras[display[-1]] + 1
                i = i + 1
                print(msg)

        counter += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
values = i
values = list(leituras.values())
expected_values = np.full(6, (i-1) / 6).tolist()

chisq, p = chisquare(values, f_exp=expected_values)

alpha = 0.05
print("O VALOR DE P É " + str(p))
if p <= alpha:
    print('DADO E VICIADO')
else:
    print('DADO NAO E VICIADO')


cv2.destroyAllWindows()
