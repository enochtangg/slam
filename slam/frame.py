import numpy 
import cv2

WIDTH = 1920//2
HEIGHT = 1080//2

class Frame():
    def __init__(self, frame):
        self.frame = cv2.resize(frame, (WIDTH, HEIGHT))
        self.orb_kp_frame = None
        self.track_kp_frame = None
        
        self.extract_features()
        self.good_features_to_track()

    def extract_features(self):
        # Initialize ORB detector and find keypoints
        orb = cv2.ORB_create()
        kps = orb.detect(self.frame, None)
        # Compute descriptors with ORB
        kps, des = orb.compute(self.frame, kps)

        # Draw keypoints onto new frame
        # print(kp)
        self.orb_kp_frame = cv2.drawKeypoints(self.frame, kps, None, color=(0,255,0), flags=0)

    def good_features_to_track(self):
        kps = []
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=100, qualityLevel=0.01, minDistance=1)
        for corner in corners:
            x, y = corner.ravel()
            kp = cv2.KeyPoint(x=x, y=y, _size=1)
            kps.append(kp)
        print(kps)
        self.track_kp_frame = cv2.drawKeypoints(self.frame, kps, None, color=(225,0,0), flags=0)

    def show(self):
        cv2.imshow('Frame', self.frame)
        cv2.imshow('Frame1', self.orb_kp_frame)
        cv2.imshow('Frame2', self.track_kp_frame)

 