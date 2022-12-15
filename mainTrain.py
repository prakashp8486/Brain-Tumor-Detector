import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten

image_directory = 'datasets/'

image_directory

no_tumor_images = os.listdir(image_directory+'no/')

yes_tumor_images = os.listdir(image_directory+'yes/')

path = 'no0.jpg'
path.split('.')[1]

dataset = []
label   = []

for i, image_name in enumerate(no_tumor_images):               # looping through no_tumor_images
    if image_name.split('.')[1]=='jpg':                        # getting only .jpg files               
        image=cv2.imread(image_directory + 'no/' + image_name) # reading images using cv.imread()
        image=Image.fromarray(image,'RGB')                     # converting image from BGR to RGB
        image=image.resize((64,64))                            # resize image
        dataset.append(np.array(image))                        # independent variables
        label.append(0)                                        # dependent variable

for i, image_name in enumerate(yes_tumor_images):                # looping through no_tumor_images
    if image_name.split('.')[1]=='jpg':                          # getting only .jpg files               
        image=cv2.imread(image_directory + 'yes/' + image_name)  # reading images using cv.imread()
        image=Image.fromarray(image,'RGB')                       # converting image from BGR to RGB
        image=image.resize((64,64))                              # resize image
        dataset.append(np.array(image))                          # independent variables
        label.append(1)                                          # dependent variable

plt.imshow(dataset[0])
plt.show()

plt.imshow(dataset[-3])
plt.show()

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train/255
X_test = X_test/255


model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(64,64,3) ) )
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_uniform' ) )
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),padding='same' ,kernel_initializer='he_uniform' ) )
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64))               # hidden layer
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))                # hidden layer
model.add(Activation('sigmoid'))   # Output layer


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

callback=keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto',
    baseline=None, restore_best_weights=True)

model.fit(X_train,
           y_train, 
           batch_size=16,
           verbose=True, 
           epochs=20, 
           validation_data=(X_test,y_test),
           shuffle=False,
           callbacks = [callback]
          )

model.save('ver1.h5')