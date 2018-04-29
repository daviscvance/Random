from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import wandb
import numpy as np
import cv2
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config
# parameters
config.batch_size = 32
config.num_epochs = 5
input_shape = (64, 64, 1)

wandb_callback=  WandbKerasCallback(save_model=False)

def load_fer2013():
    
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape
print('Samples:',num_samples)
print('Classes:',num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(4,4), padding='same',  activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.35))

model.add(Conv2D(32, kernel_size=(3,3), padding='same',  activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(4,4), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(288, activation='relu'))
model.add(Dropout(.30))
model.add(Dense(144, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(72, activation='relu'))
model.add(Dropout(.20))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.25,
        zoom_range=0.25,
	rotation_range=35,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_faces, train_emotions)
validation_generator = test_datagen.flow(val_faces, val_emotions)

model.summary()

model.fit_generator(
        train_generator,
        steps_per_epoch=2500,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=1250)


model.save("emotion.h5")


