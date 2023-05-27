import cv2
import numpy as np
import os

def preprocess_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
            img_path = os.path.join(directory, filename)
            label = int(filename.split("_")[0])  # Extract the label from the filename
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (100, 100))  # Resize the image to a consistent size
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Provide the path to the directory containing the fingerprint images
image_directory = r"C:\Users\Yusha\PycharmProjects\pythonProject2tst\image\Real"
images, labels = preprocess_images(image_directory)

def split_dataset(images, labels, test_size=0.2):
    num_samples = len(images)
    indices = np.random.permutation(num_samples)
    split = int(test_size * num_samples)

    train_indices = indices[split:]
    test_indices = indices[:split]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]

    return train_images, train_labels, test_images, test_labels

# Split the dataset into training and testing sets
train_images, train_labels, test_images, test_labels = split_dataset(images, labels)


# Flatten the 2D fingerprint images into 1D arrays
train_data = train_images.reshape(train_images.shape[0], -1)
test_data = test_images.reshape(test_images.shape[0], -1)

# Create an SVM model
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

# Train the SVM model
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Predict labels for the testing set
_, predictions = svm.predict(test_data)

# Calculate accuracy
accuracy = np.mean(predictions == test_labels) * 100
print("Accuracy: {:.2f}%".format(accuracy))
