import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Directory where the images' dataset is stored
dataset_directory = '/Users/eeshan/Downloads/dataset'

# Categories (2 : normal and abnormal)
categories = ['normal', 'abnormal']

data = []
labels = []

# Loading and preprocessing the images
for category in categories:
    category_dir = os.path.join(dataset_directory, category)

    for image_filename in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        if image is not None and not image.size == 0:
            image = cv2.resize(image, (224, 224))  # Resize to 224x224
            image = cv2.convertScaleAbs(image)  # Ensure the image is 8-bit unsigned

            data.append(image)
            labels.append(int(category == 'abnormal'))  # Label 1 for abnormal, 0 for normal

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Feature extraction using HOG with adjusted parameters
def extract_hog_features(images, n_components=32):
    hog_features = []

    # Adjusting HOG parameters to match the 224x224 image size
    hog = cv2.HOGDescriptor((224, 224), (16, 16), (8, 8), (8, 8), 9) #((window size),(block size),(step size),(cell size),(no. of bins))
    for image in images:
        hog_features.append(hog.compute(image).flatten()) #converting multi-hog to 1D array

    # Applying PCA for dimensionality reduction(HoG vectors too long for our images)
    pca = PCA(n_components=n_components) #PCA [Principal Component Analysis]:reduces dimensionality of HoG feature vectors.
    hog_features_pca = pca.fit_transform(hog_features)
    return hog_features_pca

X_train_hog = extract_hog_features(X_train, n_components=32)
X_test_hog = extract_hog_features(X_test, n_components=32)

#Labels are binary (0 or 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

# SVM Model for HOG(Binary Classification)
svm_hog_classifier = svm.SVC()
svm_hog_classifier.fit(X_train_hog, y_train)

# Predicting labels for the test set
y_pred_svm_hog = svm_hog_classifier.predict(X_test_hog)
svm_hog_accuracy = accuracy_score(y_test, y_pred_svm_hog)

#Printing the accuracy value
print(f"SVM (HOG) Accuracy: {svm_hog_accuracy}")

#Finding out confusion matrix
binary_cm = confusion_matrix(y_test, y_pred_svm_hog)

# Plotting confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Set2', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Binary Classification')

# Plotting bar chart for predicted vs. true labels
plt.figure(figsize=(8, 6))
plt.bar(['Correct Predictions', 'Incorrect Predictions'], [np.sum(y_test == y_pred_svm_hog), len(y_test) - np.sum(y_test == y_pred_svm_hog)], color=['cyan', 'magenta'])
plt.xlabel('Prediction Outcome')
plt.ylabel('Number of Samples')
plt.title('Predicted vs. True Labels')
plt.text(0, np.sum(y_test == y_pred_svm_hog), f'Accuracy: {svm_hog_accuracy:.2f}', ha='center', va='bottom', fontsize=15, color='red',fontstyle='normal')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.figure()

#Plotting the acccuracy
plt.plot(range(len(X_test_hog)), [svm_hog_accuracy] * len(X_test_hog), label='Accuracy', linestyle='dashdot',color='red')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.show()