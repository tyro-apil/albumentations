import os
import numpy as np

IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

def extract_labels(label_path):
  bboxes = []
  category_ids = []

  # Read from label text file
  with open(label_path, 'r') as file:
    lines = file.readlines()

  breakpoint()

  for line in lines:

    parts = line.strip().split()
    print(parts)
    class_num = parts[0]
    bbox_coordinates = [float(part) for part in parts[1:]]
    bbox_coordinates[2] = np.abs(bbox_coordinates[2] - 0.5 / 640)
    bbox_coordinates[3] = np.abs(bbox_coordinates[3] - 0.5 / 480)

    category_ids.append(class_num)
    bboxes.append(bbox_coordinates) 
  
  return category_ids, bboxes

def augment_image(img, transform, bboxes, category_ids, file_name):
  images_list = []          # List of string
  saved_bboxes = []         # List of list
  saved_category_ids = []   # List of list
  out_img_paths = []        # List of string
  out_label_paths = []      # List of string

  # for thrice the dataset: 2 augmentations for a single image
  for i in range(2):
    
    outfile = f'{file_name}-aug{i}'
    out_img_path = os.path.join(IMAGE_DIR, f'{outfile}.jpg')
    out_label_path = os.path.join(LABEL_DIR, f'{outfile}.txt')

    transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)

    if len(transformed["bboxes"]) == 0:
      continue

    images_list.append(transformed["image"])
    saved_bboxes.append(transformed["bboxes"])
    saved_category_ids.append(transformed["category_ids"])
    out_img_paths.append(out_img_path)
    out_label_paths.append(out_label_path)
  
  return images_list, saved_bboxes, saved_category_ids, out_img_paths, out_label_paths