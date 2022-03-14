from bottlegourd_segmentor import segment_image
import pathlib
import cv2
import tqdm


data_path = 'r'
target_path = 'e'

path = pathlib.Path(data_path)
target = pathlib.Path(target_path)

for folder in path.iterdir():
	folder_name = folder.name

	target_folder = target / folder_name
	if not target_folder.exists():
		target_folder.mkdir()

	all_files = list(folder.glob('**/*.jpg'))
	#print(all_files)

	for image_path in tqdm.tqdm(all_files, total=len(all_files), desc=folder_name):
		name, _ = image_path.name.split('.')
		# print("<<<<<<<<<<<<<<<<",name)
		image = cv2.imread(str(image_path))

		for idx, extracted_image in enumerate(segment_image(image)):
			if extracted_image is not None:
				save_name = str(target_folder / f'{name}_ {idx}.png')
				# print("<<<<<<<<<<<<<<<<",save_name)
				cv2.imwrite(save_name, extracted_image)
