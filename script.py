import cv2
import albumentations as A

import random
random.seed(7)

category_ids = [0, 1, 2]
category_id_to_name = {'blue': 0, 'purple': 1, 'red': 2}

import os
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'
img_files = os.listdir(IMAGE_DIR)
label_files = os.listdir(LABEL_DIR)

transform = A.Compose(
  [
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.2),
    A.OneOf([
      A.Blur(blur_limit=5, p=0.5),
      A.MotionBlur(blur_limit=5, p=0.8)
    ], p=1.0)
  ], bbox_params=A.BboxParams(format='yolo',  min_visibility=0.3, min_area=3600., label_fields=[])
)  


for i, img_file in enumerate(img_files):

  # Split the filename and extension
  file_name, file_extension = os.path.splitext(img_file)

  img_path = os.path.join(IMAGE_DIR, f'{file_name}.jpg')
  label_path = os.path.join(LABEL_DIR, f'{file_name}.txt')

  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  
  bboxes = []
  saved_category_ids = []

  # Read from label text file
  with open(label_path, 'r') as file:
    lines = file.readlines()

  for line in lines:

    parts = line.strip().split()
    class_num = int(parts[0])
    bbox_coordinates = [float(part) for part in parts[1:]]

    saved_category_ids.append(class_num)
    bboxes.append(bbox_coordinates) 
  
  images_list = []
  saved_bboxes = []
  
  out_img_paths = []
  out_label_paths = []

  # for thrice the dataset: 2 augmentations for a single image
  for i in range(2):
    
    outfile = f'{file_name}-aug{i}'
    out_img_path = os.path.join(IMAGE_DIR, f'{outfile}.jpg')
    out_label_path = os.path.join(LABEL_DIR, f'{outfile}.txt')

    transformed = transform(image=img, bboxes=bboxes)
    breakpoint()
    
    if len(transformed["bboxes"]) == 0:
      continue

    images_list.append(transformed["image"])
    saved_bboxes.append(transformed["bboxes"])

    out_img_paths.append(out_img_path)
    out_label_paths.append(out_label_path)
    
    

  # Write the augmented image to images directory
  for img in images_list:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img_path, img)
  
  print(saved_bboxes)

  # Write the labels to label text file and put into labels directory
  for aug_bboxes, out_label_path in zip(saved_bboxes, out_label_paths):
    with open(out_label_path, 'a') as txt_file:
      
      for category_id, aug_bbox in zip(saved_category_ids, aug_bboxes):
        labels = []
        labels.append(str(category_id))
        bbox_info = [str(info) for info in aug_bbox]
        labels += bbox_info
      
      label_txt = ' '.join(labels)

      txt_file.write(label_txt + '\n')