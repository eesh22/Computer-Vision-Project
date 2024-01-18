import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# SVM Model for HOG with reduced dimensionality (Binary Classification)
svm_hog_classifier = svm.SVC()
svm_hog_classifier.fit(X_train_hog, y_train)
y_pred_svm_hog = svm_hog_classifier.predict(X_test_hog)
svm_hog_accuracy = accuracy_score(y_test, y_pred_svm_hog)
print(f"SVM (HOG) Accuracy with PCA: {svm_hog_accuracy}")

# Confusion matrix for binary classification
#binary_cm = confusion_matrix(y_test, y_pred_svm_hog)
#plt.figure(figsize=(6, 6))
#sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.title('Binary Classification Confusion Matrix')
#plt.show()

# Classify "abnormal" images further into Mild, Moderate, Severe, and Proliferative (Multi-class)
abnormal_indices = np.where(y_test == 1)  # Assuming "abnormal" label is 1
abnormal_indices = abnormal_indices[0]  # Extract the first element of the tuple
abnormal_images = [X_test[i] for i in abnormal_indices]
abnormal_labels = [y_test[i] for i in abnormal_indices]

# SVM Model for Multi-class Classification
svm_multiclass = svm.SVC()
svm_multiclass.fit(abnormal_images, abnormal_labels)
y_pred_multiclass = svm_multiclass.predict(abnormal_images)
multiclass_accuracy = accuracy_score(abnormal_labels, y_pred_multiclass)

print(f"SVM (Multi-class) Accuracy: {multiclass_accuracy}")

# Confusion matrix for multi-class classification
multiclass_cm = confusion_matrix(abnormal_labels, y_pred_multiclass)
plt.figure(figsize=(6, 6))
sns.heatmap(multiclass_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories[1:], yticklabels=categories[1:])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Multi-class Classification Confusion Matrix')
plt.show()

# Count the number of images in each class
mild_count = np.sum(y_pred_multiclass == 0)
moderate_count = np.sum(y_pred_multiclass == 1)
severe_count = np.sum(y_pred_multiclass == 2)
proliferative_count = np.sum(y_pred_multiclass == 3)

print(f"Number of Mild Abnormalities: {mild_count}")
print(f"Number of Moderate Abnormalities: {moderate_count}")
print(f"Number of Severe Abnormalities: {severe_count}")
print(f"Number of Proliferative Abnormalities: {proliferative_count}")