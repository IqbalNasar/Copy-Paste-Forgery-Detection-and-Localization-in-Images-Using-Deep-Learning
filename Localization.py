import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pathlib
from skimage.io import imread, imshow
from PIL import Image as im
import shutil
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
img_width = 128
img_height = 128
batch_size=32
img_channels = 3
epochs=20


data_dir=os.path.join (r"C:\Users\Nasir\PycharmProjects\my\dataset")
data_dir=pathlib.Path(data_dir)

imglist=os.listdir(data_dir)
imagelen=len(imglist)


Forged=np.zeros((imagelen, img_height, img_width, img_channels))
Masked=np.zeros((imagelen, img_height, img_width, 1))



i=0
num=0
while (num*26<(imagelen)):
    for n in range(25):

      img1 = os.path.join(data_dir, imglist[26*num])
      img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
      img1 = img1 / 255.0

      newimage1 = cv2.resize(img1, (img_width, img_height))
      newimage1=np.expand_dims(newimage1, axis=-1)

      Masked[i+n]=newimage1

      img2 = os.path.join(data_dir, imglist[26 * num + (n + 1)])
      img2 = cv2.imread(img2)
      img2 = img2 / 255.0
      newimage2 = cv2.resize(img2, (img_width, img_height))
      Forged[i + n] = newimage2


    num+=2
    i+=25
Masked=Masked[0:i-25+n+1]
Forged=Forged[0:i-25+n+1]

print("The size of Masked Images is :",len(Masked))
print("The size of Forged Images is :",len(Forged))




trainimg, testimage = train_test_split(Forged, test_size=0.1, random_state=42)
trainmasked, testmasked = train_test_split(Masked, test_size=0.1, random_state=42)




inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
tf.keras.layers.BatchNormalization()
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
tf.keras.layers.BatchNormalization()
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
tf.keras.layers.BatchNormalization()
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
tf.keras.layers.BatchNormalization()
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
tf.keras.layers.BatchNormalization()
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
tf.keras.layers.BatchNormalization()
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
tf.keras.layers.BatchNormalization()
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
tf.keras.layers.BatchNormalization()
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
tf.keras.layers.BatchNormalization()
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history= model.fit(trainimg, trainmasked, batch_size=batch_size, epochs=epochs)
test_loss, test_acc = model.evaluate(testimage,  testmasked, verbose=1)
test_img_number = np.random.randint(0, len(testimage))
x= testimage[test_img_number]
test_img_input  = np.expand_dims(x, 0)
prediction = np.squeeze(model.predict(test_img_input))

print('The test acc is:',test_acc)


plt.figure(figsize=(16, 16))
ax = plt.subplot(1, 3, 1)
imshow(x)
plt.title('Actual Test Image')
plt.axis("off")
ax = plt.subplot(1, 3, 2)
imshow(testmasked[test_img_number])
plt.title('Actual Mask')
plt.axis("off")
ax = plt.subplot(1, 3, 3)
imshow(prediction)
plt.title('Predicted segmentation')
plt.axis("off")
plt.show()
