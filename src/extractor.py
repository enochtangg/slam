import numpy as np
import cv2


class FeatureExtractor:
    def __init__(self):
        self.orbs = cv2.ORB_create()

    def extract(self, img):
        # Detection
        pts = cv2.goodFeaturesToTrack(image=np.mean(img, axis=2).astype(np.uint8), maxCorners=2500,
                                      qualityLevel=0.01, minDistance=3)
        
        # Extraction
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
        kps, des = self.orbs.compute(img, kps)
        # print(kps)

        return np.array(kps), np.array(des)
