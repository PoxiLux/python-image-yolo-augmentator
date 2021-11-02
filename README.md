# Python Image (YOLO) Augmentator
Applies various augmentations to YOLO labelled (or unlabbeld) images:
- Brightness
- Gamma
- Mixup's (overlapping images, combining bounding boxes if labels are avilable)
- Vertical Flipping (both image and labels)
- Cutouts
- Noise

## Requirements
`pip install alive-progress opencv-python numpy`

## Usage
1. Place images inside a directory called `/images`.
2. Then run using your python terminal `python.exe Augmentator.py`, and watch the magic happen. 😊

![alt text](https://i.imgur.com/tSM5QcB.png)
