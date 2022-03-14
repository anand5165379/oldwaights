import cv2
import tensorflow as tf
import pathlib
import numpy as np
from scipy.spatial.distance import cdist
import math

model = tf.keras.models.load_model('paddy_unet.h5', compile=False)


def normalize(img):
    return img / 255.0


def rescale_and_pad_np_img_to(
        np_image: np.ndarray,
        target_size,
        background=(25, 55, 115)
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


def rotate_boxes(img, box, center):
    rows, cols, _ = img.shape

    length = cdist([box[1]], [box[0]])
    width = cdist([box[2]], [box[1]])
    angle = np.arctan2((box[1][1] - box[0][1]), (box[1][0] - box[0][0])) if length > width \
        else np.arctan2((box[2][1] - box[1][1]), (box[2][0] - box[1][0]))

    #     print(length[0][0], width[0][0], length[0][0] * width[0][0])

    if max(length[0][0], width[0][0]) > 700 and length[0][0] * width[0][0] < 500000:

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
    else:
        return None


def filter_contours_by_area(np_image):
    contours, hierarchy = cv2.findContours(np_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [i for i in contours if cv2.contourArea(i) > 4000]
    return contours


def get_cropped(conti, img, mult=1.5):
    rect = cv2.minAreaRect(conti)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    W = rect[1][0]
    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

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
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*mult), int(croppedH*1.05)), 
                                           (size[0]/2, size[1]/2))
    else:
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*1.05), int(croppedH*mult)), 
                                           (size[0]/2, size[1]/2))
    if croppedRotated.shape[1]>croppedRotated.shape[0]:
        croppedRotated = cv2.rotate(croppedRotated, cv2.ROTATE_90_CLOCKWISE)
    return croppedRotated


def segment_image(bgr_image) -> []:
    padding = 300
    if bgr_image is None:
        return []
    bgr_image = cv2.copyMakeBorder(
        bgr_image,
        padding, padding,
        padding, padding,
        cv2.BORDER_CONSTANT, value=(115, 55, 25)
    )
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    h, w, c = bgr_image.shape
    image = cv2.resize(rgb_image, (448, 448))
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(normalize(image)).squeeze()
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = cv2.resize(pred_mask.squeeze().astype('uint8'), dsize=(w, h))
    out_mask = pred_mask.astype('uint8')

    extracted_image_list = []
    contour_list = filter_contours_by_area(out_mask)

    for contour in contour_list:
        extracted_image = get_cropped(contour, rgb_image, mult=1.4)
        if extracted_image is not None:
            extracted_image = rescale_and_pad_np_img_to(extracted_image, (224, 448), (25, 55, 115))
            extracted_image_list.append(extracted_image)
    return extracted_image_list
