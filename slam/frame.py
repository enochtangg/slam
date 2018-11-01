import numpy 
import cv2

WIDTH = 1920//2
HEIGHT = 1080//2

class Frame():
    def __init__(self, frame):
        self.frame = cv2.resize(frame, (WIDTH, HEIGHT))
        self.kp_frame = None
        
        self.good_features_to_track()

    def good_features_to_track(self):
        kps = []
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=2000, qualityLevel=0.05, minDistance=4)
        for corner in corners:
            x, y = corner.ravel()
            kp = cv2.KeyPoint(x=x, y=y, _size=0.5)
            kps.append(kp)
        print(kps)
        self.kp_frame = cv2.drawKeypoints(self.frame, kps, None, color=(0,255,0), flags=0)

    def show(self):
        cv2.imshow('Frame', self.frame)
        cv2.imshow('Frame', self.kp_frame)
