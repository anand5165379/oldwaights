import os
import cv2
import pickle
import zipfile
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

import warnings
warnings.filterwarnings('ignore')

from dde_v2 import segment_from_np_image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class AutoExtractionSMC:

    def __init__(self):

        self.file_directory = os.path.dirname(os.path.abspath(__file__))
        self.segmentor_input_dir = os.path.join(self.file_directory, 'iter_1_raw_images')
        self.segmentor_output_dir = os.path.join(self.file_directory, 'iter_1_extractions')
        self.smc_input_dir = self.segmentor_output_dir
        self.smc_output_dir = os.path.join(self.file_directory, 'iter_1_smc_sorted')
        self.smc_classifier = load_model(os.path.join(self.file_directory, 'paddy_smc_v2.h5'),  compile=False)
        self.smc_class_names = pickle.load(open(os.path.join(self.file_directory, 'paddy_smc_v2_class_names.json'), 'rb'))['class_names']

    def RunSegmentor(self, input_image_path, output_image_path):

        for index, image in enumerate(tqdm(os.listdir(f"{input_image_path}"))):
            img = cv2.imread(f"{input_image_path}/{image}")
            
            #resizing image
            if img is not None:
            	img = cv2.resize(img, (4208, 3416))
            
            try:
                imgs = segment_from_np_image(img)
            except:
                pass
        
            if imgs is not None:
                [cv2.imwrite(f"{output_image_path}/{image}_ext{j}.jpg", cv2.cvtColor(i, cv2.COLOR_RGB2BGR)) for j, i in enumerate(imgs)] 

    def CreateDataframe(self, path):

        img_paths = []
        main_types = []
        class_labels = []

        IGNORE = []

        for imgFname in sorted(os.listdir(path)):
            if imgFname.endswith('.jpg'):
                img_paths.append(os.path.join(path,imgFname))

        data = pd.DataFrame()
        data['img_path'] = img_paths
        data = data.sample(frac=1.0, random_state=0)

        return data

    def predict_seed_types(self, im_path, dsize=(448, 224)):

        img = image.load_img(im_path, color_mode='rgb', target_size=dsize)
        img = image.img_to_array(img).astype('float32') / 255.0

        y = self.smc_classifier.predict(np.expand_dims(img, axis=0))
        y = self.smc_class_names[np.argmax(y)]
        return y

    def GenerateSegmentorOutput(self):

        list = []
        for file in os.listdir(self.segmentor_input_dir):
            #if file.endswith(".zip"):
            list.append(file.split('.')[0])

        if not os.path.isdir(self.segmentor_output_dir):
            os.mkdir(self.segmentor_output_dir)

        for variety in list:
            #variety_zip = variety + '.zip'
            #with zipfile.ZipFile(os.path.join(self.segmentor_input_dir, variety_zip), 'r') as zip_ref:
            #    zip_ref.extractall(self.segmentor_input_dir)

            if not os.path.isdir(os.path.join(self.segmentor_output_dir, variety)):
                os.mkdir(os.path.join(self.segmentor_output_dir, variety))

            input_image_path = os.path.join(self.segmentor_input_dir, variety)
            output_image_path = os.path.join(self.segmentor_output_dir, variety)

            self.RunSegmentor(input_image_path, output_image_path)

    def generate_smc_output(self):

        list = []
        for file in os.listdir(self.smc_input_dir):
            if os.path.isdir(os.path.join(self.smc_input_dir, file)):
                list.append(file)

        if not os.path.isdir(self.smc_output_dir):
            os.mkdir(self.smc_output_dir)

        for variety in tqdm(list):
            if not os.path.isdir(os.path.join(self.smc_output_dir, variety)):
                os.mkdir(os.path.join(self.smc_output_dir, variety))

            input_path = os.path.join(self.smc_input_dir, variety)
            output_path = os.path.join(self.smc_output_dir, variety)

            data = self.CreateDataframe(input_path)

            if not os.path.isdir(os.path.join(self.smc_output_dir, variety, 'good')):
                os.mkdir(os.path.join(self.smc_output_dir, variety, 'good'))

            if not os.path.isdir(os.path.join(self.smc_output_dir, variety, 'bad')):
                os.mkdir(os.path.join(self.smc_output_dir, variety, 'bad'))

            for class_ in self.smc_class_names:
                if not os.path.isdir(os.path.join(self.smc_output_dir, variety, 'bad', class_)):
                    os.mkdir(os.path.join(self.smc_output_dir, variety, 'bad', class_))

            for im_path in tqdm(data.img_path):
                smc_label = self.predict_seed_types(im_path)

                if(smc_label == 'good_seed'):
                    shutil.copy(im_path, os.path.join(self.smc_output_dir, variety, 'good'))
                elif(smc_label == 'fault1_connected'):
                    shutil.copy(im_path, os.path.join(self.smc_output_dir, variety, 'bad', 'fault1_connected'))
                elif(smc_label == 'fault2_cut'):
                    shutil.copy(im_path, os.path.join(self.smc_output_dir, variety, 'bad', 'fault2_cut'))
                elif(smc_label == 'fault3_disoriented'):
                    shutil.copy(im_path, os.path.join(self.smc_output_dir, variety, 'bad', 'fault3_disoriented'))
                elif(smc_label == 'black_seed'):
                    shutil.copy(im_path, os.path.join(self.smc_output_dir, variety, 'bad', 'black_seed'))


if __name__ == '__main__':

    auto = AutoExtractionSMC()
    auto.GenerateSegmentorOutput()
    print('sorting started. Please wait...')
    auto.generate_smc_output()
    print('sorting completed.')



        
            


