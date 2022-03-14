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


parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--output')


def annotate_image(image_path: str) -> dict:
	filename = image_path.rsplit('/', maxsplit=1)[-1]
	image = Image.open(image_path)
	image = np.array(image)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	mask = cv2.inRange(hsv, np.uint8([0, 65, 40]), np.uint8([120, 200, 225]))
	image[mask == 0] = 255

	contour_list, hierarchy = cv2.findContours(
		mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
	)

	result, contour, c_contour = [], [], []

	for cnt in contour_list:
		if any([
			len(np.squeeze(cnt).tolist()) == 2,
			cv2.contourArea(cnt) < 10000,
			cv2.contourArea(cnt) > 200000
		]):
			continue
		contour.append(np.squeeze(cnt).tolist())
		c_contour.append(cnt)

	region_list = []
	for cnt in contour:
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

	for image_path in tqdm.tqdm(path.glob('**/*.jpg')):
		filename = image_path.name.rsplit('/', maxsplit=1)[-1]
		json_to_save = annotate_image(str(path / image_path))
		with open(f"{str(out_path / filename)}.json", 'w') as fp:
			json.dump(json_to_save, fp)


if __name__ == '__main__':
	args = parser.parse_args()
	data = args.data
	output = args.output

	main(data, output)