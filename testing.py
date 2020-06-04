import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

sys.path.append("/home/fred/BFD/python/GrooveToolbox/")
from Groove import *
from LoadGrooveFromBFDPalette import *

directory = "/home/fred/BFD/python/jaki/DATASET - All Numpy Files - 10part"

folders = os.listdir(directory)
allhits = []
alltiming = []
alltempo = []

allGrooves = np.empty([0,32,5])


def _groupGroove5KitParts(groove10Parts):
    # Group kit parts into 5 polyphony levels
    # Only works for groove files - not microtiming (less important to group microtiming).
    # 0 - Kick
    # 1 - Snare
    # 2 - Closed cymbals (hihat and ride)
    # 3 - Open cymbals (open hihat, crash and extra cymbal
    # 4 - Toms (low mid and high)

    kick = 0
    snare = 1
    closedhihat = 2
    openhihat = 3
    ride = 4
    crash = 5
    extraCymbal = 6
    lowTom = 7
    midTom = 8
    highTom = 9

    groove5Parts = np.zeros([groove10Parts.shape[0], 5])
    groove5Parts[:, 0] = groove10Parts[:, kick]
    groove5Parts[:, 1] = groove10Parts[:, snare]
    groove5Parts[:, 2] = np.clip([groove10Parts[:, closedhihat] + groove10Parts[:, ride]], 0, 1)
    groove5Parts[:, 3] = np.clip(
        [groove10Parts[:, openhihat] + groove10Parts[:, crash] + groove10Parts[:, extraCymbal]], 0, 1)
    groove5Parts[:, 4] = np.clip(
        [groove10Parts[:, lowTom] + groove10Parts[:, midTom] + groove10Parts[:, highTom]], 0, 1)
    return groove5Parts

def encodeCategorical(allGrooves):
    # 0 - Kick
    # 1 - Snare
    # 2 - Closed cymbals (hihat and ride)
    # 3 - Open cymbals (open hihat, crash and extra cymbal
    # 4 - Toms (low mid and high)
    # this works. now need to make turn categorical into dummy for whole dataset the same.
    categories = ['ko', 'r', 'c', 's', 'k', 't', 'kc', 'kot', 'o', 'sc', 'kcot',
                  'ct', 'so', 'kco', 'co', 'kso', 'ks', 'kt', 'st', 'ksc', 'ksco',
                  'sco', 'ksct', 'kct', 'kst', 'sct', 'ot', 'cot', 'ksot', 'kscot',
                  'sot', 'scot']

    allCategorical = np.empty([allGrooves.shape[0],32], dtype='str')
    oneHotGrooves = np.empty([allGrooves.shape[0],32,32])

    print(allCategorical.shape)
    for j in range(allGrooves.shape[0]):
        groove = np.ceil(allGrooves[j,:,:]).astype(int)
        grooveStr = np.chararray([32], itemsize=3,unicode=True)

        for i in range(allGrooves.shape[1]):
            cat = ''
            if groove[i,0] == 1:
                cat +='k'
            if groove[i,1] == 1:
                cat +='s'
            if groove[i,2] == 1:
                cat +='c'
            if groove[i,3] == 1:
                cat +='o'
            if groove[i,4] == 1:
                cat +='t'
            if cat == '':
                cat = 'r'
            grooveStr[i] = cat
        #print(grooveStr)
        #allCategorical[j,:] = grooveStr
        oh = pd.get_dummies(grooveStr, columns=categories)
        for col in categories:
            if col not in oh.columns:
                oh[col] = 0
        print(j)
        oneHotGrooves[j,:,:] = oh.to_numpy()
    return oneHotGrooves

for i in range(len(folders)):
    bundle = np.load(directory + "/" + folders[i] + "/Hits.npy")
    #print(folders[i])
    shortBundle = np.zeros([bundle.shape[0],32,5])
    for j in range(bundle.shape[0]):
        shortBundle[j,:,:] = np.ceil(_groupGroove5KitParts(bundle[j,:,:]))
    allGrooves = np.vstack((allGrooves,shortBundle))

#print(allGrooves.shape) # mnist is (60000, 28, 28) - 60000 training images, 28x28 matrix (of pixels)

oneHotGrooves = encodeCategorical(allGrooves)

print(oneHotGrooves)
print(oneHotGrooves.shape)
print(oneHotGrooves[1,:,:])

#print(x)

# s = allGrooves[10,:,:].flatten()
# print(allGrooves[10,:,:])
# oh = pd.get_dummies(encodeCategorical(allGrooves[10,:,:]))
# print(oh)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(32,10)),
#     tf.keras.layers.Dense(128,activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
# ])

# embedding
# encoding
#

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

#output
model.add(tf.keras.layers.Dense(320, activation='softmax'))

model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
             tf.keras.callbacks.ModelCheckpoint('../models/model.h5', save_best_only=True, save_weights_only=False)]


#x_train = first bar, y_train = second bar
# need to one-hot encode drum loops. then split bar by bar
history = model.fit(X_train,  y_train,
                    batch_size=512, epochs=15,
                    callbacks=callbacks,
                    validation_data=(X_valid, y_valid))