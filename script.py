import random

import albumentations as A
import cv2

from utils import IMAGE_DIR, LABEL_DIR, augment_image, extract_labels

random.seed(7)  # set a fixed seed for reproducibility of results

category_ids = [0, 1, 2]
category_id_to_name = {"blue": 0, "purple": 1, "red": 2}

import os

img_files = os.listdir(IMAGE_DIR)
label_files = os.listdir(LABEL_DIR)

transform = A.Compose(
  [
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.2),
    A.OneOf([A.Blur(blur_limit=5, p=0.5), A.MotionBlur(blur_limit=5, p=0.8)], p=1.0),
  ],
  bbox_params=A.BboxParams(
    format="yolo", min_visibility=0.3, min_area=3600.0, label_fields=[]
  ),
)


for img_file in img_files:
  # Split the filename and extension
  file_name, file_extension = os.path.splitext(img_file)

  # Read image & convert to RGB format
  img_path = os.path.join(IMAGE_DIR, f"{file_name}.jpg")
  label_path = os.path.join(LABEL_DIR, f"{file_name}.txt")

  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Extract labels from text file
  category_ids, bboxes = extract_labels(label_path)

  # Composition of transformations
  transform = A.Compose(
    [
      A.HorizontalFlip(p=0.8),
      A.SomeOf(
        [
          A.OneOf(
            [
              A.MotionBlur(blur_limit=(3, 7), p=0.5),
              A.MedianBlur(blur_limit=(3, 7), p=0.5),
            ],
            p=0.50,
          ),
          A.OpticalDistortion(
            distort_limit=(-0.30, 0.35), shift_limit=(-0.05, 0.10), p=0.15
          ),
          A.OneOf(
            [
              A.PixelDropout(p=0.3),
              A.MultiplicativeNoise(
                multiplier=(0.55, 1.2), per_channel=True, elementwise=True, p=0.5
              ),
            ],
            p=0.66,
          ),
          A.OneOf(
            [
              A.Sharpen(alpha=(0.15, 0.50), lightness=(1.04, 2.01), p=0.33),
              A.RandomBrightnessContrast(p=0.33),
            ],
            p=0.5,
          ),
        ],
        p=1.0,
      ),
    ],
    bbox_params=A.BboxParams(
      format="yolo", min_visibility=0.33, min_area=2500.0, label_fields=["category_ids"]
    ),
  )

  # Get augmented images for a single image
  images_list, saved_bboxes, saved_category_ids, out_img_paths, out_label_paths = (
    augment_image(img, transform, bboxes, category_ids, file_name)
  )

  # Write the augmented image to images directory
  for img, out_img_path in zip(images_list, out_img_paths):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img_path, img)

  # Write the labels to label text file and put into labels directory
  for aug_bboxes, out_label_path in zip(saved_bboxes, out_label_paths):
    with open(out_label_path, "a") as txt_file:
      for category_id, aug_bbox in zip(saved_category_ids, aug_bboxes):
        labels = []
        labels.append(str(category_id))
        bbox_info = [str(info) for info in aug_bbox]
        labels += bbox_info

        label_txt = " ".join(labels)
        txt_file.write(label_txt + "\n")
