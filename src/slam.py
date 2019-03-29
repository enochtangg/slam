from frame import process_frame

import cv2


if __name__ == '__main__':
    cap = cv2.VideoCapture('../videos/test.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            process_frame(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
