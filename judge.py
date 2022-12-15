import glob
import numpy as np
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import tensorflow as tf

import os
import numpy as np


x = []
y = []
dog_breed = []

dog_breed_path = glob.glob("C:/code_temp/judge_dogs/Images/*")

for index, a in enumerate(dog_breed_path):
    picture_list = glob.glob(a+"/*")
    a = a.rsplit('/',)
    a = a[-1]
    dog_breed.append(a)

    # if index > 10: break

    for b in picture_list:
        print(b)
        buf = img_to_array(load_img(b, target_size=(100,100)))
        # image = cv2.imdecode(buf,cv2.IMREAD_UNCHANGED)
        x.append(buf)
        y.append(index)

x = np.array(x)
y = np.array(y)

print(type(x))
print(type(y))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

# x_train = x_train.reshape((len(x_train), 50, 50, 1))
# y_train = y_train.reshape((len(y_train), 50, 50, 1))
# x_train = tf.stack(x_train)
# y_train = tf.stack(y_train)
# x_test = tf.stack(x_test)
# y_test = tf.stack(y_test)
# p = list(zip(x,y))
# random.shuffle(p)
# x,y = zip(*p)

# train_rate = 0.8

# x_train = x[:int(len(x)*train_rate)]
# y_train = y[:int(len(y)*train_rate)]
# x_test = x[int(len(x)*train_rate):]
# y_test = y[int(len(y)*train_rate):]

# x_train = np.array(x_train)
# y_train = np.array(y_train)

# x_test = np.array(x_test)
# y_test = np.array(y_test)

print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

y_train = to_categorical(y_train, 120)
y_test = to_categorical(y_test, 120)

input_tensor = Input(shape=(100, 100, 3))
vgg16 = VGG16(include_top=False,weights='imagenet',input_tensor=input_tensor)

top_model=Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dense(128,activation='relu'))
top_model.add(Dropout(0.3))
top_model.add(Dense(len(dog_breed),activation='softmax'))

model = Model(inputs=vgg16.input,outputs=top_model(vgg16.output))

for layer in model.layers[:19]:
  layer.trainabile = False

model.compile(loss= 'categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),metrics=['accuracy'])

history = model.fit(x_train,y_train,batch_size=50,epochs=70,validation_data=(x_test,y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plt.plot(history.history['accuracy'], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

pred = np.argmax(model.predict(x_test[0:3]), axis=1)
print(pred)

model.summary()

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

model.save("C:/code_temp/judge_dogs/results"+'/model.h5')