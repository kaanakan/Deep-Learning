from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
import numpy as np




(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
"""

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


#Auto encoder

encoding_dim = 50

inp = Input(shape=(784,)) # 28*28


encoded_inp = Dense(128, activation='relu')(inp)
encoded_inp = Dense(64, activation='relu')(encoded_inp)
encoded_inp = Dense(50, activation='relu')(encoded_inp)

decoded_inp = Dense(64, activation='relu')(encoded_inp)
decoded_inp = Dense(128, activation='relu')(decoded_inp)
decoded_inp = Dense(784, activation='sigmoid')(decoded_inp)


auto_encoder = Model(inp, decoded_inp) # mapping input to decoded input

encoder = Model(inp, encoded_inp) # mapping input to encoded input

encoded_input = Input(shape = (50,))


auto_encoder.compile(optimizer='adam', loss='mean_squared_error')

### training

auto_encoder.fit(x_train,
				 x_train,
				 epochs = 100,
				 batch_size = 250,
				 shuffle = True,
				 validation_data = (x_test, x_test))


encoded_train_imgs = encoder.predict(x_train)

encoded_imgs = encoder.predict(x_test)


print(encoded_imgs.shape)


import matplotlib.pyplot as plt


model = Sequential()

model.add(Dense(units = 1000, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 50))
model.add(Dropout(rate = 0.3))

model.add(Dense(units = 1000, kernel_initializer = 'random_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.3))

model.add(Dense(units = 1000, kernel_initializer = 'random_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.3))

model.add(Dense(units = 10, kernel_initializer = 'random_uniform', activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


model.fit(encoded_train_imgs,
			 y_train,
			epochs = 300,
			batch_size = 250,
			shuffle = True,
			validation_data = (encoded_imgs, y_test))



score = model.evaluate(encoded_imgs, y_test, verbose=0)
print('Accuracy:', score[1])