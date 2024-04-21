import cv2
import albumentations as A
import numpy as np
from PIL import Image

image = cv2.imread('')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# list of lists in yolo format
bboxes = [[1,2,3,4]]

transform = A.Compose(
  [
    A.Resize(width=640, height=480),
    A.RandomCrop(width=640, height=480),
    A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.2),
    A.OneOf([
      A.Blur(blur_limit=0.4, p=0.5),
      A.MotionBlur(blur_limit=5, p=0.7)
    ], p=1.0)
  ], bbox_params=A.BboxParams(format="yolo", min_area=2048, min_visibility=0.3, label_fields=[])
)

images_list = [image]
saved_bboxes = [bboxes[0]]

for i in range(10):
  augmentations = transform(image=image, bboxes=bboxes)
  augmented_img = augmentations["image"]

  if len(augmentations["bboxes"]) == 0:
    continue

  images_list.append(augmented_img)
  saved_bboxes.append(augmentations["bboxes"])

for i, image in enumerate(images_list):
  cv2.imwrite(f"img-{i}", image)
