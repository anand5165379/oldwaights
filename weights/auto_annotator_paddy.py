import cv2
import numpy as np
from shapely.geometry import Polygon
import json
import glob
import os
import argparse
from PIL import Image
import pathlib
import tqdm
from scipy.interpolate import splprep, splev
import warnings

warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--output')

CONTROL_POINTS = 70
HMIN = 10
HMAX = 255


def segment(np_image):
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(hsv)
    ret, thresh = cv2.threshold(h, 10, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    return contours


def smooth_contours(contour_list):
    smoothened = []
    try:
	    for contour in contour_list:
	        x,y = contour.T
	        # Convert from numpy arrays to normal arrays
	        x = x.tolist()[0]
	        y = y.tolist()[0]
	        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
	        tck, u = splprep([x,y], u=None, s=1.0, per=2)
	        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
	        u_new = np.linspace(u.min(), u.max(), CONTROL_POINTS)
	        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
	        x_new, y_new = splev(u_new, tck, der=0)
	        # Convert it back to numpy format for opencv to be able to display it
	        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
	        smoothened.append(np.asarray(res_array, dtype=np.int32))
	    return smoothened
    except:
	    return []


def annotate_image(image_path: str) -> dict:
	filename = image_path.rsplit('/', maxsplit=1)[-1]
	image = cv2.imread(str(image_path))

	contour_list = segment(image)
	smoothened_contour = smooth_contours(contour_list)
	if len(smoothened_contour) == 0:
		print(image_path)

	region_list = []
	for cnt in smoothened_contour:
		cnt = np.squeeze(cnt)
		X = [int(x[0]) for x in cnt]
		Y = [int(x[1]) for x in cnt]
		region = {
			"region_attributes": {
				"seed_variety": 'chilli',
				'seed_type': 'chilli',
				'seed_view': 'full'
			},
			'shape_attributes': {
				'name': 'polygon',
				'all_points_x': X,
				'all_points_y': Y
			}
		}
		region_list.append(region)

	save_json = {
		f"{filename}-1": {
			"filename": filename,
			"regions": region_list,
			"file_attributes": {},
			"size": 0
		}
	}

	return save_json


def main(data_path, output_path):
	path = pathlib.Path(data)
	out_path = pathlib.Path(output_path)

	if not out_path.exists():
		out_path.mkdir()

	image_list = list(path.glob('**/*.jpg'))
	for image_path in tqdm.tqdm(image_list, total=len(image_list)):
		filename = image_path.name.rsplit('/', maxsplit=1)[-1]
		json_to_save = annotate_image(str(image_path))
		with open(f"{str(out_path / filename)}.json", 'w') as fp:
			json.dump(json_to_save, fp)


if __name__ == '__main__':
	args = parser.parse_args()
	data = args.data
	output = args.output

	main(data, output)
