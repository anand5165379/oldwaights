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
    contours = [c for c in contours if cv2.contourArea(c) > 40000]
    return contours


def get_cropped(contour, img, multiplier=1):
    padd = 109 * 2
    padd_2 = 100
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = rect[1][0] + padd
    height = rect[1][1] + padd
    all_x = [i[0] for i in box]
    all_y = [i[1] for i in box]
    padd = 109
    x1 = min(all_x) - padd - padd_2
    x2 = max(all_x) + padd + padd_2
    y1 = min(all_y) - padd - padd_2
    y2 = max(all_y) + padd + padd_2

    if max((width, height)) < 600:
        return None

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(multiplier * (x2 - x1)), int(multiplier * (y2 - y1)))
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    cropped_w = width if not rotated else height
    cropped_h = height if not rotated else width

    if cropped_h > cropped_w:
        cropped_rotated = cv2.getRectSubPix(
            cropped, (int(cropped_w * multiplier), int(cropped_h * 1)),
            (size[0] / 2, size[1] / 2)
        )
    else:
        cropped_rotated = cv2.getRectSubPix(
            cropped, (int(cropped_w * 1), int(cropped_h * multiplier)),
            (size[0] / 2, size[1] / 2)
        )

    if cropped_rotated.shape[1] > cropped_rotated.shape[0]:
        cropped_rotated = cv2.rotate(cropped_rotated, cv2.ROTATE_90_CLOCKWISE)
    return cropped_rotated


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
