#########################################################################
#  bring dataset

import csv

samples = []

# this path has original recording data sets.
path = "C:/Users/kgasb/Desktop/windows-sim/windows_sim/data/"

# bring the data from file to python memory(samples).
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split the data to train/validation/test.
from sklearn.model_selection import train_test_split
temp_samples, test_samples = train_test_split(samples, test_size=0.2)
train_samples, validation_samples = train_test_split(temp_samples, test_size=0.2)


#########################################################################
# def generator for preprocessing

import cv2
import numpy as np
from sklearn.utils import shuffle


def preprocess_image(img):
    # apply GaussianBlur
    new_img = cv2.GaussianBlur(img, (3,3), 0)
    # scale to 160x160x3
    new_img = cv2.resize(new_img,(160, 160), interpolation = cv2.INTER_AREA)
    # convert to YUV color space
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # this \\ is for windows paths 
                name1 = path+'IMG/'+batch_sample[0].split('\\')[-1]
                name2 = path+'IMG/'+batch_sample[1].split('\\')[-1]
                name3 = path+'IMG/'+batch_sample[2].split('\\')[-1]

                image1 = cv2.imread(name1)
                image2 = cv2.imread(name2)
                image3 = cv2.imread(name3)
                
                image1 = preprocess_image(image1)
                image2 = preprocess_image(image2)
                image3 = preprocess_image(image3)
                
                center_angle = float(batch_sample[3])
                images.append(image1)
                images.append(image2)
                images.append(image3)
                
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)
                
                '''
                # add fliped image(left & right)
                images.append(np.fliplr(image1))
                images.append(np.fliplr(image2))
                images.append(np.fliplr(image3))
                
                # create adjusted steering measurements for the side camera images
                angles.append(-1. * center_angle)
                angles.append(-1. *(center_angle + correction))
                angles.append(-1. *(center_angle - correction))
                '''

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)

batch_size = 32
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
test_generator = generator(test_samples, batch_size=batch_size)

#########################################################################
# build a model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

input_shape = (160, 160, 3)  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255. - .5, # normalized data
        input_shape=input_shape,
        output_shape=input_shape)) 
model.add(Cropping2D(cropping = ((65,25),(0,0)))) # take important features, and cut noising data as sky, trees and mountain.
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5)) # for preventing overfit
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5)) # for preventing overfit
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5)) # for preventing overfit
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


#########################################################################
# train a model

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)/batch_size, nb_epoch=10)

model.save('model.h5')
#########################################################################
# show history_object
import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



##########################################################################
# test the model
score = model.evaluate_generator(test_generator, steps=len(test_samples)/batch_size, max_queue_size=10, workers=1, use_multiprocessing=False)
print("test set score {}".format(score))