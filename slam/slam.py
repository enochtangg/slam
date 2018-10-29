import numpy 
import cv2
from matplotlib import pyplot


def resize_frame(frame):
    frame = cv2.resize(frame, (1920//2, 1080//2))
    return frame


# Create VideoCapture object and read from input file
# Note if we input 0, the VideoCapture will read from webcamera
cap = cv2.VideoCapture('../videos/test.mp4')

if (cap.isOpened() is False):
    print("Error opening video file")

while(cap.isOpened()):
    # Capture video frame by frame
    ret, frame = cap.read()
        
    frame = resize_frame(frame)

    # Takes grayscale of frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector and find keypoints
    orb = cv2.ORB_create()
    kp = orb.detect(frame,None)

    # Good Features To Track
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = numpy.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    # Compute descriptors with ORB
    kp, des = orb.compute(frame, kp)

    kp_overlay = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    if ret is True:
        cv2.imshow('Frame', frame)
        cv2.imshow('Frame', kp_overlay)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video object and close all frames
cap.release()
cv2.destroyAllWindows()
