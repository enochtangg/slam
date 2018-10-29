import numpy 
import cv2

WIDTH = 1920//2
HEIGHT = 1080//2

class Frame():
    def __init__(self, frame):
        self.frame = cv2.resize(frame, (WIDTH, HEIGHT))
        self.kp_frame = None
        
        self.extract_features()

    def extract_features(self):
        # Initialize ORB detector and find keypoints
        orb = cv2.ORB_create()
        kp = orb.detect(self.frame, None)

        # Compute descriptors with ORB
        kp, des = orb.compute(self.frame, kp)

        # Draw keypoints onto new frame
        self.kp_frame = cv2.drawKeypoints(self.frame, kp, None, color=(0,255,0), flags=0)

    def show(self):
        cv2.imshow('Frame', self.frame)
        cv2.imshow('Frame', self.kp_frame)

 