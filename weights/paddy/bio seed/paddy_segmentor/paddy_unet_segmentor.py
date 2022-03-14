import math

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def rescale_and_pad_np_img_to(
        np_image: np.ndarray,
        target_size,
        background=(0, 0, 0)
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


def add_padding(img, background=(0, 0, 0)):
    ht, wd, cc = img.shape
    vertical_padding = int(ht * 0.15)
    horizontal_padding = int(wd * 0.15)
    #     print(img.shape)
    img = cv2.copyMakeBorder(
        img,
        vertical_padding, vertical_padding,
        horizontal_padding, horizontal_padding,
        cv2.BORDER_CONSTANT, value=background
    )
    return img


def segment(np_image):
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(hsv)
    ret, thresh = cv2.threshold(h, 10, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_cropped(conti, img, mult=1):
    
    padd = 109 *2
    padd_2 = 100
    rect = cv2.minAreaRect(conti)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    W = rect[1][0] +padd
    H = rect[1][1] + padd
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    padd = 109
    x1 = min(Xs) - padd - padd_2
    x2 = max(Xs) + padd + padd_2
    y1 = min(Ys) - padd - padd_2
    y2 = max(Ys) + padd + padd_2

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle+=90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W
    
    if croppedH>croppedW:
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*1)), 
                                           (size[0]/2 , size[1]/2))
    else:
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*1), int(croppedH*mult)), 
                                           (size[0]/2, size[1]/2))
    if croppedRotated.shape[1]>croppedRotated.shape[0]:
        croppedRotated = cv2.rotate(croppedRotated, cv2.ROTATE_90_CLOCKWISE)
        
    if W * H < 300000:
        return None
    else:
        return croppedRotated


def segment_image(img):
    img = add_padding(img, background=(115, 55, 25))
    new_img = img.copy()
    contours = segment(img)

    img_list = []

    for conti in contours:
        if cv2.contourArea(conti) > 4000:
            minm_img = get_cropped(conti, img)

            if minm_img is not None:
                extracted_img = rescale_and_pad_np_img_to(minm_img, (224, 448), (0, 0, 0))
                extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)
                img_list.append(extracted_img)

    return img_list
