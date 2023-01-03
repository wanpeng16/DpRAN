import cv2 as cv
import numpy as np
import os
import re
from PIL import Image


def optical_flow_c(filelist):
    first_frame = np.array(Image.open(filelist[0]))
    # mask = np.zeros_like(first_frame)
    # mask[..., 1] = 255
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_RGB2GRAY)
    for fp in filelist[1:]:
        frame = np.array(Image.open(fp))
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        np.save(fp[:fp.find('.png')] + '.npy', flow)
        # # Computes the magnitude and angle of the 2D vectors
        # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # # Sets image hue according to the optical flow direction
        # mask[..., 0] = angle * 180 / np.pi / 2
        # # Sets image value according to the optical flow magnitude (normalized)
        # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # # Converts HSV to RGB (BGR) color representation
        # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # # Opens a new window and displays the output frame
        # cv.imshow("dense optical flow", rgb)
        # Updates previous frame
        prev_gray = gray


if __name__ == "__main__":
    dataset = '../../dataset/liver3'
    for patient in os.listdir(dataset):
        filelist = []
        path = os.path.join(dataset, patient)
        for fp in os.listdir(path):
            if fp.startswith('ceus') and not fp.endswith('npy'):
                filelist.append(fp)

        filelist = sorted(filelist, key=lambda x: int(re.findall(r"\d+", x)[0]))
        filelist = [os.path.join(dataset, patient, f) for f in filelist]
        optical_flow_c(filelist)
