import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
os.environ['TFDS_DATA_DIR'] = '/tmp/tensorflow_datasets'
# Load the dataset
(ds_train, ds_test), ds_info = tfds.load("ucf101", split=["train", "test"], as_supervised=True, with_info=True)

(ds_train, ds_test), ds_info = tfds.load("ucf101", split=["train", "test"], as_supervised=True, with_info=True)

# Get class names
class_names = ds_info.features["label"].names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

# Preprocessing function for video data
def preprocess(video, label):
    video = tf.image.resize(video, (224, 224)) / 255.0  # Resize and normalize
    return video, label

# Prepare dataset
ds_train = ds_train.map(preprocess).batch(16).shuffle(500)
ds_test = ds_test.map(preprocess).batch(16)

# Display sample videos from the dataset
def plot_sample_videos(dataset, title):
    plt.figure(figsize=(10, 5))
    for videos, labels in dataset.take(1):
        videos = videos.numpy()
        labels = labels.numpy()
        for i in range(min(5, len(videos))):
            ax = plt.subplot(1, 5, i + 1)
            plt.imshow(videos[i][0])  # Display the first frame of each video
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.suptitle(title)
    plt.show()

plot_sample_videos(ds_train, "Sample Training Videos")

# Define a simple 3D CNN model for deepfake detection
model = keras.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation="relu", input_shape=(None, 224, 224, 3)),
    layers.MaxPooling3D((1, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation="relu"),
    layers.MaxPooling3D((1, 2, 2)),
    layers.Conv3D(128, (3, 3, 3), activation="relu"),
    layers.MaxPooling3D((1, 2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(num_classes, activation="softmax")  # Adjust to deepfake detection classes
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(ds_train, validation_data=ds_test, epochs=5)

# Save the trained model
model.save("deepfake_detector.h5")
print("Model training completed and saved as 'deepfake_detector.h5'")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

plot_training_history(history)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Generate predictions
y_true = []
y_pred = []
for videos, labels in ds_test.take(10):
    preds = model.predict(videos)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

plot_confusion_matrix(y_true, y_pred, class_names)
