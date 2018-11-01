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
        orb = cv2.ORB_create()
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(image=gray, maxCorners=2000, qualityLevel=0.05, minDistance=4)
        
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=5) for pt in pts]
        kps, des = orb.compute(self.frame, kps)
        self.kp_frame = cv2.drawKeypoints(self.frame, kps, None, color=(0,255,0), flags=0)

    def show(self):
        cv2.imshow('Frame', self.frame)
        cv2.imshow('Frame', self.kp_frame)
