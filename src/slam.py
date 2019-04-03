from frame import Frame

import cv2


if __name__ == '__main__':
    # loading video into VideoCapture obj
    cap = cv2.VideoCapture('../videos/test.mp4')
    frame = Frame(None)
    while cap.isOpened():
        # ret -> boolean that states whether it succesfully got an image
        # img -> a single img in the video
        ret, img = cap.read()

        if ret is True:
            # send image into Frame object to be processed
            frame.process_frame(img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
