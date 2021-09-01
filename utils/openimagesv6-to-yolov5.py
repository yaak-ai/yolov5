import pandas as pd
from pathlib import Path
from tqdm import tqdm

import argparse

def SaveBoundingBoxToFile(image_root, annotations_root, image_id, label, x_min, x_max, y_min, y_max):
	# Check that the image exist:
	if image_root.joinpath(image_id).exists():

		# If the label file exist, append the new bounding box
		ann_file = annotations_root.joinpath(image_id).with_suffix('.txt')
		ann_file.parent.mkdir(parents=True, exist_ok=True)
		modifier = 'a' if ann_file.exists() else 'w'
		with ann_file.open(modifier) as f:
				f.write(' '.join([str(trainable_codes.index(label)),
		                    str(round((x_max+x_min)/2, 6)),
		                    str(round((y_max+y_min)/2, 6)),
		                    str(round(x_max-x_min, 6)),
		                    str(round(y_max-y_min, 6))])+'\n')


if __name__ == '__main__':

	parser = argparse.ArgumentParser("OpenImagesV6 to YOLOv5 format converter")
	parser.add_argument('-i', '--images-dir', help="Root path for images", type=Path)
	parser.add_argument('-f', '--annotation-files', nargs='+', help="OpenImagesV6 annotation files", type=Path)
	parser.add_argument('-c', '--classes-file', help="OpenImagesV6 class description file", type=Path)
	parser.add_argument('-a', '--annotations-dir', help="Root path for YOLOV5 annotations", type=Path)

	args = parser.parse_args()

	# Get the codes for the trainable classes
	class_descriptions = pd.read_csv(args.classes_file.resolve().as_posix(), names=["class_code", "class_name"], header=None)
	trainable_codes = [code for code, name in class_descriptions.values]
	# trainable_codes = [code for code,name in class_descriptions.values] # For ALL CLASSES

	for filename in args.annotation_files:

		# Read the train da
		# filename = "train-annotations-bbox.csv"
		df = pd.read_csv(filename.resolve().as_posix())

		# Keep only the data for our training labels
		# Comment this line for ALL CLASSES
		df = df.loc[df['LabelName'].isin(trainable_codes)]

		tqdm.pandas(desc=f"Mapping OpenImagesV6 annotations in {filename} to YOLOV5")

		# Save the bounding box data to the files
		df.progress_apply(lambda x: SaveBoundingBoxToFile(args.images_dir, args.annotations_dir, x['ImageID'], x['LabelName'], x['XMin'], x['XMax'], x['YMin'], x['YMax']), axis=1)
