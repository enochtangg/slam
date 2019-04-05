from extractor import FeatureExtractor
from display import Display

import cv2
import numpy as np


# IRt = np.eye(4)


# Frame is a wrapper class used by main script to call extractor and display.
class Frame:
    def __init__(self, img):
        self.feature_extractor = FeatureExtractor()
        self.display = Display()
        self.frames = []

        self.kpus = None
        self.des = None
        self.pts = None

        if img is not None:
            self.kpus, self.des = self.feature_extractor.extract(img)
            self.pts = [None]*len(self.kpus)

    def match_frames(self, f1, f2):
        # Matching previous frame and current frame (used in triangulation
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            matches = bf.knnMatch(f1.des, f2.des, k=2)
        except cv2.error:
            return False

        good_matches = []
        idx_of_des_from_f1, idx_of_des_from_f2 = [], []
        idx_set1, idx_set2 = set(), set()

        # using Lowe's ratio to filter out bad kps
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                p1 = f1.kpus[m.queryIdx]
                p2 = f2.kpus[m.trainIdx]

                # ensure distance is within 32
                if m.distance < 32:
                    if m.queryIdx not in idx_set1 and m.trainIdx not in idx_set2:
                        idx_of_des_from_f1.append(m.queryIdx)
                        idx_of_des_from_f2.append(m.trainIdx)
                        idx_set1.add(m.queryIdx)
                        idx_set2.add(m.trainIdx)

                        good_matches.append((p1, p2))

        print(good_matches)

        # now we are left with good matches
        # we want see all the points are are inliers with ransac
        # then return the index_ofde

        return matches

    def process_frame(self, img):
        img = cv2.resize(img, (self.display.W, self.display.H))

        # takes image from cap and turns it into a frame & gets kps/des
        frame = Frame(img)
        self.frames.append(frame)
        if len(self.frames) <= 1:
            return

        # implement some matching frame function that takes the last two frames from self.frames
        pts, Rt = self.match_frames(self.frames[-1], self.frames[-2])
        # cv2.triangulatePoints(IRt, Rt, pts[:,0].T, pts[:,1].T).T

        # for 2D display
        kps_frame = self.display.process_kps_to_frame(img, frame.kpus)
        self.display.show(img)
        self.display.show(kps_frame)
