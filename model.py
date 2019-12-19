#!/usr/bin/python3

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import math
import numpy as np
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import jit, cuda, float32
from IPython.display import HTML
import csv
from sklearn.utils import shuffle
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Activation, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from pathlib import Path

import argparse
from group_norm import GroupNormalization
from platform import python_version
import threading

print(python_version())

def getModel(shape, loss='mse', optimizer='adam'):
    # https://devblogs.nvidia.com/deep-learning-self-driving-cars/ 
    NModel = Sequential()

    # Normalization Layer
    NModel.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=shape))
    
    # Convolutional Layer 1
    NModel.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    # Group normalization module for keras.  
    # Downloaded from https://github.com/titu1994/Keras-Group-Normalization
    NModel.add(GroupNormalization(groups=12, axis=-1))
    NModel.add(Activation('relu'))

    # Convolutional Layer 2
    NModel.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    NModel.add(GroupNormalization(groups=18, axis=-1))
    NModel.add(Activation('relu'))

    # Convolutional Layer 3
    NModel.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    NModel.add(GroupNormalization(groups=12, axis=-1))
    NModel.add(Activation('relu'))

    # Convolutional Layer 4
    NModel.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    NModel.add(GroupNormalization(groups=32, axis=-1))
    NModel.add(Activation('relu'))

    # Convolutional Layer 5
    NModel.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    NModel.add(GroupNormalization(groups=32, axis=-1))
    NModel.add(Activation('relu'))

    # Flatten Layers
    NModel.add(Flatten())

    # Fully Connected Layer 1
    NModel.add(Dense(100))
    NModel.add(Activation('relu'))

    # Fully Connected Layer 2
    NModel.add(Dense(50))
    NModel.add(Activation('relu'))

    # Fully Connected Layer 3
    NModel.add(Dense(10))
    NModel.add(Activation('relu'))

    # Output Layer
    NModel.add(Dense(1))

    NModel.compile(loss=loss, optimizer=optimizer)

    return NModel

# Read all rows from all CSV files 
def readRowsFromCSV(csvFileList):
    allRows = []
    shapeSampled = False
    height, width, channels = (0, 0, 0)
    for i, csvFile in enumerate(csvFileList):
        (csvDir, tail) = os.path.split(csvFile)
        csvFp = open(csvFile)
        csvReader = csv.reader(csvFp, delimiter=',')
        for row in csvReader:
            if len(row) < 7: 
                continue
            row[0] = os.path.join(csvDir, row[0])
            row[1] = os.path.join(csvDir, row[1])
            row[2] = os.path.join(csvDir, row[2])
            allRows.append(row)
            if shapeSampled==False:
                sampleImage = cv2.imread(row[0])
                height, width, channels = sampleImage.shape
                shapeSampled = True
        csvFp.close()
    return allRows, (height, width, channels)
 
# Training sample generator for generating training data on the fly without loading all
# data into memory
def trainSamplesGenerator(trainCSVRows, maxBatchSize=1024, loadAll=False):
    nSamples = len(trainCSVRows)
    while True:
        if loadAll==True:
            images = []
            angles = []
            print("loadAll=True, Loading all sample")

        for offset in range(0, nSamples, maxBatchSize//3):
            if loadAll==False:
                images = []
                angles = []
            for i in range(maxBatchSize//3):
                idx = offset+i
                if idx >= len(trainCSVRows):
                    break
                row = trainCSVRows[idx]
                centerFileName  = row[0]
                leftFileName    = row[1]
                rightFileName   = row[2]

                if os.path.exists(centerFileName) == False:
                    print ("File not found:" + centerFileName)
                if os.path.exists(leftFileName) == False:
                    print ("File not found:" + leftFileName)
                if os.path.exists(rightFileName) == False:
                    print ("File not found:" + rightFileName)
                    
                if os.path.exists(centerFileName) == True and os.path.exists(leftFileName) == True and os.path.exists(rightFileName) == True:
                    steeringCenter  = float(row[3])
                    throttle    = float(row[4])
                    breakOn     = float(row[5])
                    speed       = float(row[6])

                    # Small adjustment to shift the car a little bit to the left to avoid 
                    # hitting poles at some sharp turns on track 2 
                    centerCorrection = -0.06
                
                    # Adjustment to left image
                    leftCorrection = 0.22  
                    # Adjustment to right image is a little bit more to shift the car 
                    # to the left
                    rightCorrection = 0.24 

                    steeringCenter += centerCorrection
                    steeringLeft  = steeringCenter + leftCorrection
                    steeringRight = steeringCenter - rightCorrection

                    centerImage = Image.open(centerFileName)
                    leftImage   = Image.open(leftFileName)
                    rightImage  = Image.open(rightFileName)
                    width   = centerImage.size[0]
                    height  = centerImage.size[1]
                    channels = 3
                    nparrayCenter   = np.asarray(centerImage, dtype="uint8")[:,:,:channels]
                    nparrayLeft     = np.asarray(leftImage, dtype="uint8")[:,:,:channels]
                    nparrayRight    = np.asarray(rightImage, dtype="uint8")[:,:,:channels]

                    images.append(nparrayCenter)
                    angles.append(steeringCenter)
                    # Discard over-steering samples
                    if steeringLeft >= -1.0 and steeringLeft <= 1.0: 
                        images.append(nparrayLeft)
                        angles.append(steeringLeft)
        
                    if steeringRight >= -1.0 and steeringRight <= 1.0: 
                        images.append(nparrayRight)
                        angles.append(steeringCenter)

            if loadAll==False:
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

        if loadAll==True:
            print (str(len(images)) + " samples loaded")
            for offset in range(0, nSamples, maxBatchSize):
                if offset + maxBatchSize < nSamples:
                    X_train = np.array(images[offset:offset+maxBatchSize])
                    y_train = np.array(angles[offset:offset+maxBatchSize])
                else:
                    X_train = np.array(images[offset:nSamples])
                    y_train = np.array(angles[offset:nSamples])
                yield shuffle(X_train, y_train)



def main():
    parser = argparse.ArgumentParser(description='Train a car to drive itself')
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        type=str,
        default='./processed_data',
        help='Root directory of driving data samples'
    )
    parser.add_argument(
        '--epoch',
        dest='epoch',
        type=int,
        default=30,
        help='Number of epoch to train'
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=512,
        help='Batch size'
    )


    args = parser.parse_args()

    csvFileList = []
    for filename in Path(args.data_dir).glob('**/driving_log.csv'):
        print("Adding CSV file: " + str(filename))
        csvFileList.append(str(filename))

    csvRowList, imageShape = readRowsFromCSV(csvFileList)
    print("Number of training data loaded: {:5d}".format(len(csvRowList)*3))
    # Shuffle the samples 4 times for even distribution
    shuffledRowList = shuffle(csvRowList)
    shuffledRowList = shuffle(shuffledRowList)
    shuffledRowList = shuffle(shuffledRowList)
    shuffledRowList = shuffle(shuffledRowList)
    # Use 70% of the samples as training set; 30% as validation set
    nRow = len(csvRowList)
    nTrain = int(nRow * 0.7)
    nValid = nRow - nTrain
    batchSize = args.batch_size
    train_generator = trainSamplesGenerator(shuffledRowList[0:nTrain], batchSize, loadAll=False)
    validation_generator = trainSamplesGenerator(shuffledRowList[nTrain:nRow], batchSize, loadAll=False)

    model = getModel(imageShape, loss='mse', optimizer='adam')

    with open('model.summary', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.close()
    print(model.summary())
    
    plot_model(model, show_shapes=True, to_file='model.png') 

    modelFile = 'model.h5'
    history = model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(nTrain/batchSize), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(nValid/batchSize), \
            epochs=args.epoch, verbose=1)
    print("Model saved: "+ modelFile)
    model.save(modelFile)

    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_trend.png', bbox_inches='tight')
    
if __name__ == '__main__':
    main()
