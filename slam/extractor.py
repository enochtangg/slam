import numpy 
import cv2

class FeatureExtractor():
    def __init__(self):
        self.orbs = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.last = None

    def extract(self, img):
        # Detection
        pts = cv2.goodFeaturesToTrack(image=numpy.mean(img, axis=2).astype(numpy.uint8), maxCorners=2500, qualityLevel=0.01, minDistance=3)
        
        # Extraction
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
        kps, des = self.orbs.compute(img, kps)

        # Matching
        matches = None
        if self.last is not None:
            matches = self.bf.match(des, self.last['des'])
            print(matches)
        self.last = { 'kps': kps, 'des': des }

        return kps, des, matches