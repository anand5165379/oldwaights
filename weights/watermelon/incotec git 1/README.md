## Installing the extractor
```shell
pip3 install git+https://github.com/agdhi/crop_segmentors.git@incotech
```

### How to use

Edit `run_segmentor.py` and give the `data_path`, `target_path` and `variety`
`data_path` is where the variety folders should be and should follow the directory structure:
```
+ data_path
	+ variety-1
		- image_1.jpg
		- image_2.jpg
		- image_3.jpg
	+ variety-2
		- image_4.jpg
		- image_5.jpg
		- image_6.jpg
```
`target_path` can be empty

After giving the paths, open a terminal in the folder where the script is and execute: `python3 run_segmentor.py`
