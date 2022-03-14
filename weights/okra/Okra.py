import time
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import math
import cv2
import os
from imutils import perspective, paths
from scipy.spatial import distance as dist
from PIL import Image
from tqdm import tqdm


class Okra_Seg(object):
    def __init__(self, hsv_h=(20, 255), hsv_s=(0, 255), hsv_v=(0, 255), size=(224, 448, 3)):
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
        hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv)
#         ret, thresh = cv2.threshold(h, 200, 255, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(h, self.hsv_h_low, self.hsv_h_high, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_list = []
        contours = [i for i in contours if cv2.contourArea(i) > 500]
        return contour_list, contours, thresh
    
    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
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
    
    def execute_optimised_watermelon(self, img):
        img = self.add_padding(img)
        new_img = img.copy()
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        conts, contours, _ = self.segment(img)

        img_list_pre = []
        img_list = []

        for conti in contours:
            rect = cv2.minAreaRect(conti)
            box = np.int0(cv2.boxPoints(rect))
            box *= 5
            box = perspective.order_points(box)
            center = tuple([i * 5 for i in rect[0]])

#             for (x, y) in box:
#                 (tl, tr, br, bl) = box
#                 (tltrX, tltrY) = self.midpoint(tl, tr)
#                 (blbrX, blbrY) = self.midpoint(bl, br)
#                 (tlblX, tlblY) = self.midpoint(tl, bl)
#                 (trbrX, trbrY) = self.midpoint(tr, br)
#                 dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#                 dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#                 print(dB, dA)
#             if dB < 1500 and dA < 1400:
            minm_img, length_mm, width_mm, length_mm_width_mm = self.rotate_boxes(new_img, box, center)
            if minm_img is not None:
                img_list_pre.append(minm_img)
            
        for minm_img in img_list_pre:
            extracted_img = self.rescale_and_pad_np_img_to(minm_img, (self.size[0], self.size[1]), (115, 55, 25))
            extracted_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2RGB)
            img_list.append(extracted_img)
        
        return img_list
    
def RunSegmentor(input_folder_path,output_folder_path, save_type='plt', save_format='jpg', size=(224, 448, 3)):
    """
        input_folder_path: Input folder having all raw images
        output_folder_path: Output folder where extracted images are to be saved
        save_type: save with matplotlibs PIL based write method or cv2 method
        save_format: jpg, jpeg, png 
        size: size of extracted image, defaults to (224, 448, 3)
    
    """
    cseg = Okra_Seg((170, 240), (0, 255), (0, 255), size)
    for index, image in enumerate(tqdm(os.listdir(f"{input_folder_path}"))):
        img = cv2.imread(f"{input_folder_path}/{image}")
        try:
            img_list = cseg.execute_optimised_watermelon(img)
            if img_list is not None:
                if save_type == 'plt':
                    [plt.imsave(f"{output_folder_path}/{image}_ext{j}.{save_format}", i) for j, i in enumerate(img_list)]
                elif save_type == 'cv2':
                    [cv2.imwrite(f"{output_folder_path}/{image}_ext{j}.{save_format}", cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for j, i in enumerate(img_list)]
                else:
                    print("Not a supported save type : Use plt or cv2 instead")
        except:
             print(f"Error in {input_folder_path}/{image}")