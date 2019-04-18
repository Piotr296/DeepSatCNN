import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential

# https://www.kaggle.com/crawford/deepsat-sat4
# _____________________________________________________________________________
# - Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared.
# - X_train_sat4.csv: 400,000 training images, 28x28 images each with 4 channels
# - y_train_sat4.csv: 400,000 training labels, 1x4 one-hot encoded vectors
# - X_test_sat4.csv: 100,000 training images, 28x28 images each with 4 channels
# - y_test_sat4.csv: 100,000 training labels, 1x4 one-hot encoded vectors

IM_SIZE = (28, 28, 4)
X_TRAIN_PATH = 'X_train_sat4.csv'
Y_TRAIN_PATH = 'y_train_sat4.csv'
X_TEST_PATH = 'X_test_sat4.csv'
Y_TEST_PATH = 'y_test_sat4.csv'
class_names = ['Barren Land', 'Trees', 'Grassland', 'None']


def index(arr):
    ind = arr.tolist().index(1)
    return ind


# Load data and labels
X_train = pd.read_csv(X_TRAIN_PATH, nrows=10000)
Y_train = pd.read_csv(Y_TRAIN_PATH, nrows=10000)

X_test = pd.read_csv(X_TRAIN_PATH, nrows=100)
Y_test = pd.read_csv(Y_TRAIN_PATH, nrows=100)

# Convert pandas to numpy
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

# Now we have to reshape each of them from a list of numbers to a 28*28*4 image and normalize dataset
X_train_img = X_train.reshape(-1, *IM_SIZE).astype(np.uint8)/255
X_test_img = X_test.reshape(-1, *IM_SIZE).astype(np.uint8)/255

# Check some picture
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_img[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[index(Y_train[i])])
plt.show()

# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 4)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train_img,
          Y_train,
          validation_data=(X_test_img, Y_test),
          epochs=3)

# we can make prediction for test data
prediction = model.predict(X_test_img)
print(prediction)
print(np.argmax(prediction[0]))
print(index(Y_test[0]))

plt.figure()
plt.imshow(X_test_img[0])
plt.xlabel(class_names[index(Y_test[0])])
plt.colorbar()
plt.grid(False)
plt.show()



