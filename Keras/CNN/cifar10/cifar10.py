from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam

"""
========================================================================
"""

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train/255.0

Y_train = to_categorical(Y_train)

X_test = X_test/255.0

Y_test = to_categorical(Y_test)

"""
========================================================================
"""
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

model.summary()

"""
========================================================================
"""


model.fit(X_train, Y_train,
              batch_size=128,
              shuffle=True,
              epochs=120,
              validation_data=(X_test, Y_test))


"""
========================================================================
"""

scores = model.evaluate(X_test , Y_test)
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

"""
	Loss: 0.706
	Accuracy: 0.815
"""
