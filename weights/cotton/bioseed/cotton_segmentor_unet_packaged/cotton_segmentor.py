import math
import tensorflow as tf

from skimage.transform import resize
from skimage.measure import label as LABEL
from skimage.morphology import opening, square
from skimage.measure import regionprops as REGION
import cv2
import numpy as np
from imutils import perspective
from scipy.spatial.distance import cdist


model = tf.keras.models.load_model('cotton_segmentor_unet.h5', compile=False)


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


def add_padding(img, background):
    ht, wd, cc = img.shape
    # create new image of desired size and color (blue) for padding
    padding_vert = int(ht * 0.5)
    padding_hor = int(wd * 0.5)

    img = cv2.copyMakeBorder(
        img,
        padding_vert, padding_vert,
        padding_hor, padding_hor,
        cv2.BORDER_CONSTANT, value=background
    )

    return img


def segment(np_image):
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)
    ret, thresh = cv2.threshold(h, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_cropped(conti, img, mult=2):
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
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*2), int(croppedH*1.4)), 
                                           (size[0]/2, size[1]/2))
    else:
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*1.4), int(croppedH*2)), 
                                           (size[0]/2, size[1]/2))
    if croppedRotated.shape[1]>croppedRotated.shape[0]:
        croppedRotated = cv2.rotate(croppedRotated, cv2.ROTATE_90_CLOCKWISE)
    return croppedRotated


def mask_to_bbox(image, mask):
    img_list = []
    mask = opening(mask, square(4))
    sequenced_mask = LABEL(mask)
    concern = sequenced_mask*255
    concern[concern>255] = 255
    concern = np.expand_dims(concern,axis =2)

    fy = image.shape[0]/512
    fx = image.shape[1]/512
    ret, thresh = cv2.threshold(np.array(concern,dtype=np.uint8), 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh,None,fx = fx,fy = fy)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c,conti in enumerate(contours):
        minm_img = get_cropped(conti, image)
        extracted_img = rescale_and_pad_np_img_to(minm_img, (224, 224), (25, 55, 115))

        img_list.append(extracted_img)
    return img_list


def segment_image(img):
    padded = add_padding(img, background=(25, 55, 115))
    resized = cv2.resize(padded, (512, 512)) / 255.0

    out = model.predict(np.expand_dims(resized, axis=0))[0]
    mask = (out[:, :, 1] > 0.6) * ~(out[:, :, 2] > 0.2)

    img_list = mask_to_bbox(padded, mask)
    return img_list
