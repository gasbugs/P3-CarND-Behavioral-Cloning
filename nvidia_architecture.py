import csv
import cv2
import numpy as np

lines = []
path = "C:/Users/kgasb/Desktop/windows-sim/windows_sim/data/"
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = path + 'IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

argumented_images, argumented_measurements = [], []
for image, measurement in zip(images, measurements):
    argumented_images.append(image)
    argumented_measurements.append(measurement)
    argumented_images.append(cv2.flip(image,1))
    argumented_measurements.append(measurement * -1.0)
    

    
X_train = np.array(argumented_images)
y_train = np.array(argumented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


print('X_train.shape:',X_train.shape)
print('y_train.shape:',y_train.shape)

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, verbose = 2,
          shuffle=True, nb_epoch=7)

model.save('model.h5')



####################################################################3
from numpy import random 

def train_generator(features, labels, batch_size=128):
 # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 160,320,3))
    batch_labels = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator(X_train,y_train,128), 
                                     steps_per_epoch=10000,
                                     nb_epoch=5, verbose=1)

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