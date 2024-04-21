import cv2
import albumentations as A
import numpy as np
from PIL import Image

image = Image.open("")

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
  ]
)

images_list = [image]
image = np.array(image)

for i in range(10):
  augmentations = transform(image=image)
  augmented_img = augmentations["image"]
  images_list.append(augmented_img)