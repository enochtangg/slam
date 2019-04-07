import cv2
import numpy as np


class Display:
    def __init__(self):
        self.W = 1920//2
        self.H = 1080//2

    @staticmethod
    def process_kps_to_frame(img, kps_frame):
        img2 = cv2.drawKeypoints(img, kps_frame, np.array([]), color=(255, 0, 0), flags=0)
        return img2

    @staticmethod
    def show(img):
        cv2.imshow('Frame', img)
