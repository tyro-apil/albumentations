# Encountered errors
---
1. **Unable to put additional arguments like category_ids (class ids) in the transform function obtained from A.Compose**
  
    I was putting class_labels as the class_ids for all the possible classes present
    ```
    [0,1,2]
    #Note: {0: 'blue', 1: 'purple', 2: 'red'}
    ```
    But class_labels are for the class_ids of the given bounding boxes i.e. 
    >Suppose there are two bounding boxes within an image say red, then purple. For the given image, class_labels must be:
    ``` [2,1] ```

    ***Solution***: [official_docs for bbox augmentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)

2. **Yolo bboxes out of bound after augmentation by albumentations**
    
    ***Solution***: [github_issues](https://github.com/albumentations-team/albumentations/issues/862)
