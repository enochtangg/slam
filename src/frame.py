from extractor import FeatureExtractor
from display import Display
from helpers import EssentialMatrixTransform, fundamentalToRt

import cv2
import numpy as np

from skimage.measure import ransac, LineModelND

IRt = np.zeros((3,4))
IRt[:, :3] = np.eye(3)

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

    @staticmethod
    def match_frames(f1, f2):
        # Matching previous frame and current frame (used in triangulation
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            matches = bf.knnMatch(f1.des, f2.des, k=2)
        except cv2.error:
            return False

        good_matches = []
        idx_of_des_from_f1, idx_of_des_from_f2 = [], []
        idx_set1, idx_set2 = set(), set()

        # using Lowe's ratio to filter out bad matches
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

        good_matches = np.array(good_matches)
        idx_of_des_from_f1 = np.array(idx_of_des_from_f1)
        idx_of_des_from_f2 = np.array(idx_of_des_from_f2)


        # # fit matrix
        # model, inliers = ransac((good_matches[:, 0], good_matches[:, 1]), EssentialMatrixTransform, min_samples=8,
        #                         residual_threshold=0.02, max_trials=100)
        #
        # print(model)

        # return idx_of_des_from_f1[inliers], idx_of_des_from_f2[inliers], fundamentalToRt(model.params)
        return good_matches, idx_of_des_from_f1, idx_of_des_from_f2

    def process_frame(self, img):
        img = cv2.resize(img, (self.display.W, self.display.H))

        # takes image from cap and turns it into a frame & gets kps/des
        frame = Frame(img)
        self.frames.append(frame)
        if len(self.frames) <= 1:
            return

        f1 = self.frames[-1]
        f2 = self.frames[-2]

        # implement some matching frame function that takes the last two frames from self.frames
        good_matches, idx1, idx2 = self.match_frames(f1, f2)

        proj_points_1 = np.array([np.array([kp.pt[0] for kp in good_matches[:, 0]]),
                                  np.array([kp.pt[1] for kp in good_matches[:, 0]])])

        proj_points_2 = np.array([np.array([kp.pt[0] for kp in good_matches[:, 1]]),
                                  np.array([kp.pt[1] for kp in good_matches[:, 1]])])
        
        # figure out how to get camera matrix 2 (Rt)
        points_in_3d = cv2.triangulatePoints(IRt, Rt, proj_points_1, proj_points_2)
        print(points_in_3d)

        # for 2D display
        kps_frame = self.display.process_kps_to_frame(img, frame.kpus)
        self.display.show(img)
        self.display.show(kps_frame)
