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

original=np.zeros((imagelen, img_height, img_width, img_channels))
Forged=np.zeros((imagelen, img_height, img_width, img_channels))
Masked=np.zeros((imagelen, img_height, img_width, img_channels))



i=0
num=0
while (num*26<(imagelen)):
    for n in range(25):

      img1 = os.path.join(data_dir, imglist[26*num])
      img1 = cv2.imread(img1)
      img1 = img1 / 255.0
      newimage1 = cv2.resize(img1, (img_width, img_height))
      Masked[i+n]=newimage1

      img2 = os.path.join(data_dir, imglist[26 * num + (n + 1)])
      img2 = cv2.imread(img2)
      img2 = img2 / 255.0
      newimage2 = cv2.resize(img2, (img_width, img_height))
      Forged[i + n] = newimage2

      img3 = os.path.join(data_dir, imglist[26 * num + (n + 27)])
      img3 = cv2.imread(img3)
      img3 = img3 / 255.0
      newimage3 = cv2.resize(img3, (img_width, img_height))
      original[i + n] = newimage3
    num+=2
    i+=25
Masked=Masked[0:i-25+n+1]
Forged=Forged[0:i-25+n+1]
original=original[0:i-25+n+1]
print("The size of Masked Images is :",len(Masked))
print("The size of Forged Images is :",len(Forged))
print("The size of Original Images is :",len(original))

#window_name = 'image'
#cv2.imshow(window_name, Masked[25])
#cv2.waitKey(0)

images=np.concatenate((original, Forged))
print("The size of concentated Images is :",(len(images)))

labels=np.zeros(len(images))
labels[0:int(len(images)/2)-1]=0
labels[int(len(images)/2):len(images)-1]=1
classes=['Original','Forged']
classlength=len(classes)
#plt.figure(figsize=(10,10))
#for t in range(25):
    #plt.subplot(5,5,t+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(images[t+9475])

#plt.show()
#print(labels[4875])

trainimg, testimage = train_test_split(images, test_size=0.1, random_state=42)
trainlabel, testlabel = train_test_split(labels, test_size=0.1, random_state=42)




inputs = tf.keras.layers.Input((img_height, img_width, img_channels))

c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.4)(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


p4=tf.keras.layers.Flatten()(p3)
p5=tf.keras.layers.Dense(512, activation='relu')(p4)
tf.keras.layers.BatchNormalization()
p6 = tf.keras.layers.Dense(512, activation='relu')(p5)
tf.keras.layers.BatchNormalization()
p7 = tf.keras.layers.Dense(128, activation='relu')(p6)
tf.keras.layers.BatchNormalization()


outputs = tf.keras.layers.Dense(1, activation='sigmoid')(p7)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()




history= model.fit(trainimg, trainlabel, batch_size=batch_size, epochs=epochs,validation_data=(testimage, testlabel))

predictions = model.predict(testimage)


test_img_number = np.random.randint(0, len(testimage))
test_loss, test_acc = model.evaluate(testimage,  testlabel, verbose=1)
print("The test accuracy is:", test_acc)
score = tf.nn.softmax(predictions[test_img_number])
print('Preictions about Forgery:',classes[np.argmax(score)])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

