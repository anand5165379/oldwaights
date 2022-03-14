import os
import cv2
import numpy as np
from PIL import Image
import shutil
from scipy.spatial.distance import cdist
import time
from imutils import paths
from matplotlib import pyplot as plt
import random
import math
from tqdm import tqdm

class Maize_Segmentor(object):
    def __init__(self, hsv_h, hsv_s, hsv_v, size):
        self.hsv_h_low = hsv_h[0]
        self.hsv_h_high = hsv_h[1]
        self.hsv_s_low = hsv_s[0]
        self.hsv_s_high = hsv_s[1]
        self.hsv_v_low = hsv_v[0]
        self.hsv_v_high = hsv_v[1]
        self.size = size
    def rescale_and_pad_np_img_to(self,
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

    def add_padding(self, img):
        ht, wd, cc= img.shape
    #     print(img.shape)

        # create new image of desired size and color (blue) for padding
        ww = int(wd + wd*0.3)

        hh = int(ht + ht*0.3)
    #     print(ww,hh)
        color = (115, 55, 25)
        result = np.full((hh,ww,cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy+ht, xx:xx+wd] = img
        return result

    def segment(self, np_image):
        hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        ret, thresh_h = cv2.threshold(h, self.hsv_h_low, self.hsv_h_high, cv2.THRESH_BINARY_INV)
        ret, thresh_s = cv2.threshold(s, self.hsv_s_low, self.hsv_s_high, cv2.THRESH_BINARY_INV)
        ret, thresh_v = cv2.threshold(v, self.hsv_v_low, self.hsv_v_high, cv2.THRESH_BINARY_INV)
        
        contours, hierarchy = cv2.findContours(thresh_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_list = []
        contours = [i for i in contours if cv2.contourArea(i) > 1000]
        return contour_list, contours, [thresh_h, thresh_s, thresh_v]
    
    
    def rotate_boxes(self, img, box, center, multiplication_factor = 0.008207485226526593):
        rows, cols, _ = img.shape

        length = cdist([box[1]], [box[0]]) 
        width = cdist([box[2]], [box[1]]) 

        length_mm = max(length[0][0], width[0][0]) * multiplication_factor
        width_mm = min(length[0][0], width[0][0]) * multiplication_factor

        angle = np.arctan2((box[1][1] - box[0][1]), (box[1][0] - box[0][0])) if length > width\
        else np.arctan2((box[2][1] - box[1][1]), (box[2][0] - box[1][0])) 

#         print(length[0][0], width[0][0], length[0][0] * width[0][0])

        if max(length[0][0], width[0][0]) > 0 and length[0][0] * width[0][0] < 50000000:
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

            affine_img = cv2.warpAffine(img, rot, (rows + 1000,cols + 1000))
            minm_img = affine_img[
                        max(0, y1 - padd) : y2 + padd,
                        max(0, x1 - padd) : x2 + padd
                    ]
#             plot(minm_img)
            minm_img = minm_img.astype(np.uint8)
            # check w/h ratio
            h, w, _ = minm_img.shape
            w_h_ratio = w / h

            if not (w_h_ratio > 5 or w_h_ratio < 0.2):
                return minm_img, length_mm, width_mm, length_mm/width_mm
        else:
            return None, length_mm, width_mm, length_mm/width_mm
        
    def execute_optimised(self, img):
        """
            Parameters:
                numpy array representing the image

            Returns:
                image_list with each item in the list being a tuple of length 3
                image_list[0] contains img, length_mm, width_mm
        """
        img = self.add_padding(img)
        new_img = img.copy()
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        conts, contours, thresh = self.segment(img)
    #     print(len(contours))
        img_list = []
        box_list = []
#         print(len(contours))
        for conti in contours:
            rect = cv2.minAreaRect(conti)
            box = np.int0(cv2.boxPoints(rect))
            box = box * 5
            center = tuple([i * 5 for i in rect[0]])
    #         print(box)
            minm_img, length_mm, width_mm, l_b_ratio = self.rotate_boxes(new_img, box, center)
            if minm_img is not None:
                extracted_img = self.rescale_and_pad_np_img_to(minm_img, self.size, (115, 55, 25))
                extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)
                img_list.append((extracted_img, length_mm, width_mm, l_b_ratio))

        return img_list
    
def RunSegmentor(input_folder_path,output_folder_path, save_type='plt', save_format='jpg', size=(224, 448)):
    """
        Inputs: 
                - input_folder_path
                - output_folder_path
                - save_type: save as 'plt' or 'cv2'
                - save_format: save as 'jpg' or 'png' or other cv2 and plt supported formats
                - size: size of extracted images, defaults to 224, 448
    """
    cseg = Maize_Segmentor((20, 255), (0, 255), (0, 255),size)
    for index, image in enumerate(tqdm(os.listdir(f"{input_folder_path}"))):
        img = cv2.imread(f"{input_folder_path}/{image}")
        img_list = cseg.execute_optimised(img)
        if img_list is not None:
            if save_type == 'plt':
                [plt.imsave(f"{output_folder_path}/{image}_ext{j}.{save_format}", i[0]) for j, i in enumerate(img_list)]
            elif save_type == 'cv2':
                [cv2.imwrite(f"{output_folder_path}/{image}_ext{j}.{save_format}", cv2.cvtColor(i[0], cv2.COLOR_BGR2RGB)) for j, i in enumerate(img_list)]
            else:
                print("Not a supported save type : Use plt or cv2 instead")