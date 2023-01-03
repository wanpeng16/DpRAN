import cv2
import operator
import numpy as np
import sys
import os
import skimage.io as scio
from skimage.feature import graycomatrix, graycoprops
import random
USE_TOP_ORDER = True

USE_LOCAL_MAXIMA = False




class Frame:
    """class to hold information about each frame

    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def extractFrames(videopath, path, k, start_time, end_time, NUM_TOP_FRAMES=24):
    cap = cv2.VideoCapture(videopath)
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fps_video * start_time + random.randint(0, fps_video // 2))
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    label_path = path + k + '\\detect.png'
    BW = scio.imread(label_path)
    BW = cv2.normalize(BW, None, 0, 1, cv2.NORM_MINMAX)
    while success and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_time * fps_video:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        luv = np.multiply(frame, BW)
        glcm = graycomatrix(
            luv, [
                2, 4, 8, 16], [
                0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
        temp = graycoprops(glcm, 'dissimilarity')

        curr_frame = temp
        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(cap.get(cv2.CAP_PROP_POS_FRAMES), diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        success, frame = cap.read()
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if USE_TOP_ORDER:
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)

    # save all keyframes as image
    cap = cv2.VideoCapture(videopath)
    keyframe_id_set = list(keyframe_id_set)
    keyframe_id_set.sort()
    for idx in keyframe_id_set:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        rval, frame = cap.read()
        name = str(idx) + ".jpg"
        cv2.imwrite(path + k + '\\keyframes_1\\' + name, frame)

    cap.release()
