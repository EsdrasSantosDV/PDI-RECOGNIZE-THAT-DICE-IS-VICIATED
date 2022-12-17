
import cv2
import numpy as np
from sklearn import cluster
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name


# Setting up the blob detector
params = cv2.SimpleBlobDetector_Params()


# images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
params.filterByColor = False
params.blobColor = 0
params.minRepeatability = 5
# Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
params.filterByArea = True
params.minArea = 3. # Highly depending on image resolution and dice size
params.maxArea = 400. # float! Highly depending on image resolution.

params.filterByCircularity = True
params.minCircularity = 0. # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
params.maxCircularity = 3.4028234663852886e+38 # infinity.

params.filterByConvexity = False
params.minConvexity = 0.
params.maxConvexity = 3.4028234663852886e+38

params.filterByInertia = True # a second way to find round blobs.
params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat
params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

params.minThreshold = 0 # from where to start filtering the image
params.maxThreshold = 255.0 # where to end filtering the image
params.thresholdStep = 5 # steps to go through
params.minDistBetweenBlobs = 3.0 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution!
params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.

detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        # Important to set min_sample to 0, as a dice may only have one dot
        clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


cap = cv2.VideoCapture('video9.mp4')

# Check if camera opened successfully
if (cap.isOpened( )== False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # We'll define these later
        blobs = get_blobs(frame)
        dice = get_dice_from_blobs(blobs)
        out_frame = overlay_info(frame, dice, blobs)
        # Display the resulting frame
        cv2.imshow('Frame' ,frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

