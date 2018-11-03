from extractor import FeatureExtractor
from display import Display

import numpy 
import cv2

WIDTH = 1920//2
HEIGHT = 1080//2

feature_extractor = FeatureExtractor()
display = Display()

def process_frame(img):
    frame = cv2.resize(img, (WIDTH, HEIGHT))
    kps, des, matches = feature_extractor.extract(frame)
    kps_frame = display.process_kps_to_frame(frame, kps)
    display.show(frame)
    display.show(kps_frame)
