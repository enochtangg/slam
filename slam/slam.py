from matplotlib import pyplot
from frame import Frame

import numpy 
import cv2

# Create VideoCapture object and read from input file
# Note if we input 0, the VideoCapture will read from webcamera
cap = cv2.VideoCapture('../videos/test.mp4')

if (cap.isOpened() is False):
    print("Error opening video file")

while(cap.isOpened()):
    # Capture video frame by frame
    ret, frame = cap.read()

    display_frame = Frame(frame)

    if ret is True:
        display_frame.show()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video object and close all frames
cap.release()
cv2.destroyAllWindows()
