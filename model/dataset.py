from model import config
import torch
from torch.utils.data import Dataset
import cv2
import os


class ImageDataset(Dataset):
    # initialize the constructor
    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        # retrieve annotations from stored list
        filename, startX, startY, endX, endY, label = self.data[index]
        startX = int(startX)
        startY = int(startY)
        endX = int(endX)
        endY = int(endY)

        # get full path of filename
        image_path = os.path.join(config.IMAGES_PATH, label, filename)

        # load the image (in OpenCV format), and grab its dimensions
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # scale bounding box coordinates relative to dimensions of input image
        startX /= w 
        startY /= h 
        endX /= w 
        endY /= h

        # normalize label in (0, 1, 2) and convert to tensor
        label = torch.tensor(config.LABELS.index(label))

        # apply image transformations if any
        if self.transforms:
            image = self.transforms(image)

        # return a tuple of the images, labels, and bounding box coordinates
        # maybe it's 2x2 tensor and not 1x4
        return image, label, torch.tensor([startX, startY, endX, endY])

    def __len__(self):
        # return the size of the dataset
        return len(self.data)
