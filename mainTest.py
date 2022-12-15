import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('ver1.h5')
image = cv2.imread('datasets/pred9.jpg')

img = Image.fromarray(image, 'RGB')
img = img.resize((64,64))

img = np.array(img)

img = img/255
img = np.expand_dims(img, axis=0)
img.shape

res = model.predict(img)
print('Brain tumor = 0:"No"\t1:"Yes"\nAnswer = ',int(res))