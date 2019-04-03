import cv2


class Display:
    def __init__(self):
        self.W = 1920//2
        self.H = 1080//2

    @staticmethod
    def process_kps_to_frame(img, kps_frame):
        img2 = cv2.drawKeypoints(img, kps_frame, None, color=(0, 255, 0), flags=0)
        return img2

    @staticmethod
    def show(img):
        cv2.imshow('Frame', img)
