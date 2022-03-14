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
    ret, thresh = cv2.threshold(s, 110, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [i for i in contours if cv2.contourArea(i) > 70000 and cv2.contourArea(i) < 170000]

    contour_list = []
    return contour_list, contours


def segment_from_np_image(img):
    img = add_padding(img)
    new_img = img.copy()
    #img = cv2.resize(img, None, fx=0.2, fy=0.2)
    conts, contours = segment(img)

    img_list = []
    box_list = []

    for conti in contours:
        x, y, w, h = cv2.boundingRect(conti)
        minm_img = new_img[y - 50 : y + h + 50, x - 50 : x + w + 50, : ]

        if minm_img is not None:
            minm_img = rescale_and_pad_np_img_to(minm_img, (224, 224))
            extracted_img = cv2.cvtColor(minm_img, cv2.COLOR_BGR2RGB)
            img_list.append(extracted_img)

    return img_list
