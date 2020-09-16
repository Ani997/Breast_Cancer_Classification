import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator        # Generates batches of image tensor data with real time data augmentation
from keras.models import Sequential, Model                             # Configures model for training
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D                   # Convolution kernel,Max Pooling kernel
from keras.layers import Activation, Dropout, Flatten, Dense    # Activation layer,Dropout layer,flattening,dense layer
from keras import backend as K    
from keras.callbacks import CSVLogger                              # Using Tensorflow backend 
from livelossplot.keras import PlotLossesCallback
import efficientnet.keras as efn
import numpy as np                                              # General purpose array processing package
from keras.preprocessing import image
                    

TRAINING_LOGS_FILE = "D:/Machine Learning/AI Projects/Breast_Cancer_Classification/logs/training_logs.csv"
MODEL_SUMMARY_FILE = "D:/Machine Learning/AI Projects/Breast_Cancer_Classification/logs/model_summary.txt"

# dimensions of our images
img_width, img_height = 300, 300                                # Height and width of input image

train_data_dir = 'D:/Machine Learning/AI Projects/Breast_Cancer_Classification/data/train'         # Training data directory
validation_data_dir = 'D:/Machine Learning/AI Projects/Breast_Cancer_Classification/data/validate' # Validation data directory
train_samples = 6141                 # No of Training samples
validation_samples = 1758            # No of Validation samples
epochs = 20                          # No of epochs
batch_size = 20                      # No of samples to be passed to NN

if K.image_data_format() == 'channels_first':         # Checking the image format
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,                               # rescaling factor 
    shear_range=0.2,                                # shear intensity
    zoom_range=0.2,                                 # range for random zoom
    horizontal_flip=True)                           # randomly flips inputs horizontally

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))         # input layer,32 filters of 3x3 kernel size
model.add(Activation('relu'))                                  # using relu activation function
model.add(MaxPooling2D(pool_size=(2,2)))                       # max pooling

model.summary()                                                # prints summary representation of the model

model.add(Conv2D(64, (3, 3)))                                  # 1st hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))                                  # 2nd hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())                                           # flattening the model
model.add(Dense(64))                                           
model.add(Activation('relu'))
model.add(Dropout(0.5))                                        # adding dropout
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
with open(MODEL_SUMMARY_FILE,"w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))


model.fit_generator(
    train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps= validation_samples // batch_size,
    callbacks=[PlotLossesCallback(), CSVLogger(TRAINING_LOGS_FILE,
                                            append=False,
                                            separator=";")], 
    verbose=1)

model.save('D:/Machine Learning/AI Projects/Breast_Cancer_Classification/model/CNN.h5')