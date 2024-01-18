import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

# Step 1: Data Collection and Preprocessing
def load_data(data_dir):
    data = []
    labels = []

    # List of categories (folder names)
    categories = os.listdir(data_dir)

    for category in categories:
        label = category  # Assuming the folder names represent the DR severity
        category_path = os.path.join(data_dir, category)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(label)

    return data, labels

# Set your dataset directory on the C drive
data_dir = r'/Users/eeshan/Desktop/PC/miniproject fcv/sample'

data, labels = load_data(data_dir)

# Step 2: Feature Extraction - HoG and SURF
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    return features

def extract_surf_features(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return descriptors

# Extract HoG and SURF features for all images
hog_features = [extract_hog_features(image) for image in data]
surf_features = [extract_surf_features(image) for image in data]

# Step 3: Traditional Machine Learning Model (SVM)
# Split the data into training and testing sets
X_train_hog, X_test_hog, X_train_surf, X_test_surf, y_train, y_test = train_test_split(hog_features, surf_features, labels, test_size=0.2, random_state=42)

# Train SVM classifiers
hog_svm = svm.SVC()
hog_svm.fit(X_train_hog, y_train)

surf_svm = svm.SVC()
surf_svm.fit(X_train_surf, y_train)

# Evaluate the SVM classifiers
hog_svm_predictions = hog_svm.predict(X_test_hog)
surf_svm_predictions = surf_svm.predict(X_test_surf)

# Step 4: Convolutional Neural Network (CNN)
# Define the categories as integer labels
categories = os.listdir(data_dir)

category_to_label = {category: label for label, category in enumerate(categories)}
labels = [category_to_label[category] for category in labels]

# Build a CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add more convolutional and pooling layers as needed
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')  # Number of classes for DR severity
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the data for the CNN
data_cnn = np.array(data).reshape(-1, 128, 128, 1)
data_cnn = data_cnn / 255.0  # Normalize pixel values

# Split the data into training and testing sets
X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(data_cnn, labels, test_size=0.2, random_state=42)

# Train the CNN
model.fit(X_train_cnn, y_train, epochs=5)

# Evaluate the CNN
cnn_predictions = model.predict(X_test_cnn)
cnn_predictions = np.argmax(cnn_predictions, axis=1)

# Map integer labels back to category names for reporting
predicted_categories = [categories[prediction] for prediction in cnn_predictions]
true_categories = [categories[label] for label in y_test]

print("CNN Classification Report:")
print(classification_report(true_categories, predicted_categories))

from sklearn.metrics import accuracy_score, precision_score

# Calculate accuracy for the SVM classifiers
hog_svm_accuracy = accuracy_score(y_test, hog_svm_predictions)
surf_svm_accuracy = accuracy_score(y_test, surf_svm_predictions)

# Calculate precision for the SVM classifiers
hog_svm_precision = precision_score(y_test, hog_svm_predictions, average='weighted')
surf_svm_precision = precision_score(y_test, surf_svm_predictions, average='weighted')

# Calculate accuracy and precision for the CNN model
cnn_accuracy = accuracy_score(y_test, cnn_predictions)
cnn_precision = precision_score(y_test, cnn_predictions, average='weighted')

# Print accuracy and precision for the SVM classifiers
print("HOG-SVM Accuracy:", hog_svm_accuracy)
print("HOG-SVM Precision:", hog_svm_precision)

print("SURF-SVM Accuracy:", surf_svm_accuracy)
print("SURF-SVM Precision:", surf_svm_precision)

# Print accuracy and precision for the CNN model
print("CNN Accuracy:", cnn_accuracy)
print("CNN Precision:", cnn_precision)

cv2.imshow()
cv2.waitKey()