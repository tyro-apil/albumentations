import os
import random
import shutil


def split_dataset(images_dir, labels_dir, train_percent, valid_percent, test_percent):
  # Create train, valid, and test directories
  train_dir = os.path.join(images_dir, "train")
  valid_dir = os.path.join(images_dir, "valid")
  test_dir = os.path.join(images_dir, "test")
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(valid_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)

  # Get the list of image files
  image_files = os.listdir(images_dir)

  # Shuffle the image files
  random.shuffle(image_files)

  # Calculate the number of images for each set
  num_images = len(image_files)
  num_train = int(num_images * train_percent)
  num_valid = int(num_images * valid_percent)

  # Split the image files into train, valid, and test sets
  train_files = image_files[:num_train]
  valid_files = image_files[num_train : num_train + num_valid]
  test_files = image_files[num_train + num_valid :]

  # Move the image files to the corresponding directories
  for file in train_files:
    shutil.move(os.path.join(images_dir, file), os.path.join(train_dir, file))
    shutil.move(
      os.path.join(labels_dir, file.replace(".jpg", ".txt")),
      os.path.join(train_dir, file.replace(".jpg", ".txt")),
    )
  for file in valid_files:
    shutil.move(os.path.join(images_dir, file), os.path.join(valid_dir, file))
    shutil.move(
      os.path.join(labels_dir, file.replace(".jpg", ".txt")),
      os.path.join(valid_dir, file.replace(".jpg", ".txt")),
    )
  for file in test_files:
    shutil.move(os.path.join(images_dir, file), os.path.join(test_dir, file))
    shutil.move(
      os.path.join(labels_dir, file.replace(".jpg", ".txt")),
      os.path.join(test_dir, file.replace(".jpg", ".txt")),
    )
