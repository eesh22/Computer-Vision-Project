import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Load the CSV file with image paths and numerical labels (0, 1, 2, 3)
df = pd.read_csv('trainLabels.csv')

# Extract image file paths and numerical labels
image_paths = df['Image_Path'].tolist()
labels = df['Label'].tolist()

# Define the number of classes
num_classes = 4  # 4 classes: Normal, Mild, Moderate, Severe

# Load and preprocess images
data = []
image_dir = r'/Users/eeshan/Desktop/PC/miniproject fcv/sample'  # Adjust the directory path to your dataset

for image_path in image_paths:
    image = cv2.imread(os.path.join(image_dir, image_path))
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    data.append(image)

# Convert numerical labels to one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=num_classes)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

# Define and compile the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))
model.add(layers.MaxPooling2D((2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(np.array(X_train), np.array(y_train), epochs=10)

# Evaluate the CNN model on the test set
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))

print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
