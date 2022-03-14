from paddy_unet_segmentor import segment_image
import pathlib
import cv2
import tqdm


data_path = '../../r'
target_path = '../../e1'

path = pathlib.Path(data_path)
target = pathlib.Path(target_path)

for folder in path.iterdir():
	folder_name = folder.name

	target_folder = target / folder_name
	if not target_folder.exists():
		target_folder.mkdir()

	all_files = list(folder.glob('**/*.jpg'))

	for image_path in tqdm.tqdm(all_files, total=len(all_files), desc=folder_name):
		name, _ = image_path.name.split('.')
		image = cv2.imread(str(image_path))

		if image is not None:
			for idx, extracted_image in enumerate(segment_image(image)):
				if extracted_image is None:
					print(image_path)
				else:
					save_name = str(target_folder / f'{name}_{idx}.png')
					extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_RGB2BGR)
					cv2.imwrite(save_name, extracted_image)
