import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()

training_images, test_images = training_images / 255.0, test_images / 255.0

class_names = ["Plane", "car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]



training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
# model.add(layers.Softmax())

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))

# loss, accuracy = model.evaluate(test_images, test_labels)
# print(f"Loss: {loss}"
#       f"\nAccuracy: {accuracy}")

# model.save("image_classifier.keras")


def prepare_image(img):
    # 1. Check if image is valid
    if img is None:
        raise ValueError("⚠️ Input image is None")
    
    # 2. Convert BGR (OpenCV default) → RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # 3. Resize to 32x32 (CIFAR-10 resolution)
    img = cv.resize(img, (32, 32))
    
    # 4. Normalize (0-1)
    img = img / 255.0
    
    # 5. Expand dimensions (batch of 1)
    img = np.expand_dims(img, axis=0)
    
    return img


#Using Model
# Prepare it for prediction
model = models.load_model("image_classifier.keras")

img = cv.imread("C:\\Users\\Abdullah\\Downloads\\cat.jpg")
img_ready = prepare_image(img)

# Predict
prediction = model.predict(img_ready)
index = np.argmax(prediction)

# Show image with label
plt.imshow(cv.cvtColor(cv.imread(r"C:\Users\Abdullah\Downloads\catcl.jpg"), cv.COLOR_BGR2RGB))
plt.title(f"Prediction: {class_names[index]}")
plt.axis("off")
plt.show()

print(f"This is a {class_names[index]}")



