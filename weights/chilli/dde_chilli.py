import cv2
import cv2
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import os
from scipy.spatial.distance import cdist
import math


def rescale_and_pad_np_img_to(
        np_image: np.ndarray,
        target_size,
        background=(115, 55, 25)
) -> np.ndarray:
    w_t, h_t = target_size
    h, w, c = np_image.shape

    if w / h < w_t / h_t:
        new_w = h * w_t / h_t
        padding = new_w - w
        image = cv2.copyMakeBorder(
            np_image,
            0, 0,
            int(padding // 2), int(padding // 2),
            cv2.BORDER_CONSTANT, value=background
        )
    else:
        new_h = w * h_t / w_t
        padding = new_h - h
        image = cv2.copyMakeBorder(
            np_image,
            int(padding // 2), int(padding // 2),
            0, 0,
            cv2.BORDER_CONSTANT, value=background
        )
    return cv2.resize(image, target_size)


def add_padding(img):
    ht, wd, cc = img.shape
    #     print(img.shape)

    # create new image of desired size and color (blue) for padding
    ww = int(wd + wd * 0.3)

    hh = int(ht + ht * 0.3)
    #     print(ww,hh)
    color = (115, 55, 25)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy + ht, xx:xx + wd] = img
    return result


def segment(np_image):
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ret, thresh = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [i for i in contours if cv2.contourArea(i) > 3000 and cv2.contourArea(i) < 20000]
    contour_list = []
    return contour_list, contours

def rotate_boxes(img, box, center):
    rows, cols, _ = img.shape

    length = cdist([box[1]], [box[0]])
    width = cdist([box[2]], [box[1]])
    angle = np.arctan2((box[1][1] - box[0][1]), (box[1][0] - box[0][0])) if length > width \
        else np.arctan2((box[2][1] - box[1][1]), (box[2][0] - box[1][0]))

    #     print(length[0][0], width[0][0], length[0][0] * width[0][0])

#     if max(length[0][0], width[0][0]) > 700 and length[0][0] * width[0][0] < 500000:

    rot = cv2.getRotationMatrix2D(center, math.degrees(angle) + 90, 1)
    box = np.int0(cv2.transform(np.array([box]), rot))[0]

    # Corner points
    # (x1, y1)--------------------(x2, y1)
    #    |        (corner pts)        |
    # (x2, y1)--------------------(x2, y2)
    x1 = min([box[0][0], box[1][0], box[2][0], box[3][0]])
    y1 = min([box[0][1], box[1][1], box[2][1], box[3][1]])
    x2 = max([box[0][0], box[1][0], box[2][0], box[3][0]])
    y2 = max([box[0][1], box[1][1], box[2][1], box[3][1]])

    padd = max([rows, cols]) // 50

    affine_img = cv2.warpAffine(img, rot, (rows + 1000, cols + 1000))
    minm_img = affine_img[
               max(0, y1 - padd): y2 + padd,
               max(0, x1 - padd): x2 + padd
               ]
    #     plot(minm_img)
    minm_img = minm_img.astype(np.uint8)
    # check w/h ratio
    h, w, _ = minm_img.shape
    w_h_ratio = w / h

    if not (w_h_ratio > 5 or w_h_ratio < 0.2):
        return minm_img


def segment_from_np_image(img):
    img = add_padding(img)
    new_img = img.copy()
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    conts, contours = segment(img)

    img_list = []
    box_list = []
    
    for conti in contours:
        rect = cv2.minAreaRect(conti)
        box = np.int0(cv2.boxPoints(rect))
        box = box * 5
        center = tuple([i * 5 for i in rect[0]])
        min_img = rotate_boxes(new_img, box, center)
        min_img = rescale_and_pad_np_img_to(min_img, (224, 224))
        img_list.append(min_img)
    return img_list
