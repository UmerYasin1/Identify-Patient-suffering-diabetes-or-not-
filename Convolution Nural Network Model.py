from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation , Dropout , Dense , Flatten
from keras import backend as K
import numpy as np
from keras_preprocessing import image


img_width, img_height = 320, 240
train_data_dir = 'Data/train'
validation_data_dir = 'Data/validation'

nb_train_sample = 50
nb_validation_sample = 10

epochs = 5
batch_size = 2

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.5,
    zoom_range=0.2,
    horizontal_flip=True,

)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = test_datagen.flow_from_directory(
    'Data/validation',
    target_size=(img_width,  img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'Data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'

)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3),))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_sample,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_sample,
)

model.save_weights('first_try.h5')

img_pred = image.load_img('Data/test/patinet2.png',
                          target_size=(320, 240))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)


result = model.predict(img_pred)
print(result)
if result[0][0] == 1:
    prediction = 'Patients'

else:
    prediction = 'Healthy'

print(prediction)
