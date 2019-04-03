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

    # def match_frames(self, kps, des):
    #     # Matching
    #     matches = None
    #     if self.last is not None:
    #         matches = cv2.BFMatcher().match(des, self.last['des'])
    #     self.last = {'kps': kps, 'des': des}
    #
    #     return matches

    def process_frame(self, img):
        img = cv2.resize(img, (self.display.W, self.display.H))

        # takes image from cap and turns it into a frame & gets kps/des
        frame = Frame(img)
        self.frames.append(frame)
        if len(self.frames) <= 1:
            return

        # implement some matching frame function that takes the last two frames from self.frames
        # pts, Rt = self.match_frames(self.frames[-1], self.frames[-2])
        # cv2.triangulatePoints(IRt, Rt, pts[:,0].T, pts[:,1].T).T

        # for 2D display
        kps_frame = self.display.process_kps_to_frame(img, frame.kpus)
        self.display.show(img)
        self.display.show(kps_frame)
