import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Directory where your image data is stored
dataset_directory = '/Users/eeshan/Desktop/PC/miniproject fcv/colored_images'  # Replace with the actual path

# List of categories (0, 1, 2, 3, 4)
categories = ['0', '1', '2', '3', '4']

data = []
labels = []

# Load and preprocess images
for category in categories:
    category_dir = os.path.join(dataset_directory, category)

    for image_filename in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        if image is not None and not image.size == 0:
            image = cv2.resize(image, (224, 224))  # Resize to 224x224
            image = cv2.convertScaleAbs(image)  # Ensure the image is 8-bit unsigned

            data.append(image)
            labels.append(int(category))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Feature extraction using OpenCV HOG with adjusted parameters
def extract_hog_features(images, n_components=32):
    hog_features = []
    # Adjust HOG parameters to match the 224x224 image size
    hog = cv2.HOGDescriptor((224, 224), (16, 16), (8, 8), (8, 8), 9)
    for image in images:
        hog_features.append(hog.compute(image).flatten())

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    hog_features_pca = pca.fit_transform(hog_features)
    return hog_features_pca

X_train_hog = extract_hog_features(X_train, n_components=32)
X_test_hog = extract_hog_features(X_test, n_components=32)

# SVM Model for HOG with reduced dimensionality
svm_hog_classifier = svm.SVC()
svm_hog_classifier.fit(X_train_hog, y_train)
y_pred_svm_hog = svm_hog_classifier.predict(X_test_hog)
svm_hog_accuracy = accuracy_score(y_test, y_pred_svm_hog)
print(f"SVM (HOG) Accuracy with PCA: {svm_hog_accuracy}")
print