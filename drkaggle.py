import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Dense,Flatten
from keras import optimizers
from keras.applications import VGG16

import cv2
import os
import random
import itertools
from glob import iglob
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

BASE_DATASET_FOLDER = '/Users/eeshan/Desktop/PC/miniproject fcv/data'
VALIDATION_FOLDER = '/Users/eeshan/Desktop/PC/miniproject fcv/data/validation'
TEST_FOLDER = '/Users/eeshan/Desktop/PC/miniproject fcv/data/test'
TRAIN_FOLDER = '/Users/eeshan/Desktop/PC/miniproject fcv/data/training'

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

IMAGE_SIZE = (224,224)
INPUT_SHAPE = (224,224, 3)

TRAIN_BATCH_SIZE =80
VAL_BATCH_SIZE = 15
EPOCHS = 50
LEARNING_RATE = 0.0001

rain_datagen = ImageDataGenerator(

    rescale=1. / 255,
    # featurewise_center=False,
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # rotation_range=45,
    # zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    # vertical_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER),
        target_size=IMAGE_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)

val_datagen=ImageDataGenerator(rescale=1./255)
val_generator=val_datagen.flow_from_directory(
    os.path.join(BASE_DATASET_FOLDER, VALIDATION_FOLDER),
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    shuffle=False)