import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

sys.path.append("/home/fred/BFD/python/GrooveToolbox/")
from Groove import *
from LoadGrooveFromBFDPalette import *

directory = "/home/fred/BFD/python/jaki/DATASET - NOT SWUNG"
np.set_printoptions(threshold=np.inf,precision=2)

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
    return np.random.shuffle(oneHotGrooves)

for i in range(len(folders)):
    bundle = np.load(directory + "/" + folders[i] + "/Hits.npy")
    #print(folders[i])
    shortBundle = np.zeros([bundle.shape[0],32,5])
    for j in range(bundle.shape[0]):
        shortBundle[j,:,:] = np.ceil(_groupGroove5KitParts(bundle[j,:,:]))
    allGrooves = np.vstack((allGrooves,shortBundle))

#print(allGrooves.shape) # mnist is (60000, 28, 28) - 60000 training images, 28x28 matrix (of pixels)

#oneHotGrooves = encodeCategorical(allGrooves)
#np.save("One-Hot-Grooves-Nonswung.npy", oneHotGrooves)
oneHotGrooves = np.load("One-Hot-Grooves-Nonswung.npy")


print(oneHotGrooves.shape)
bar1 = oneHotGrooves[0:6000,0:16,:]
bar2 = oneHotGrooves[0:6000,16:32,:]

bar1Validate = oneHotGrooves[6000:6628,0:16,:]
bar2Validate = oneHotGrooves[6000:6628,16:32,:]

X_train = bar1
y_train = bar2
X_valid = bar1Validate
y_valid = bar2Validate


# for i in range(32):
#     print(oneHotGrooves[1,i,:])

#np.save("One-Hot-Grooves.npy", oneHotGrooves)

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
hiddenDim = 24

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(200, input_shape=(16,32))) #encoder
model.summary()

model.add(tf.keras.layers.RepeatVector(16))
model.add(tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(200,))) #decoder
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='softmax')))

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(16, return_sequences=True,batch_input_shape=(512,16,32)))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))

#output
#model.add(tf.keras.layers.Dense(16, activation='softmax'))
# model.add(tf.keras.layers.LSTM(16, return_sequences=True,
#                                dropout=0.1, recurrent_dropout=0.1, input_shape=(64,)))

model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25),
             tf.keras.callbacks.ModelCheckpoint('../models/model.h5', save_best_only=True, save_weights_only=False)]

#x_train = first bar, y_train = second bar
history = model.fit(X_train,  y_train,
                    batch_size=64, epochs=500,
                    callbacks=callbacks,
                    validation_data=(bar1Validate,bar2Validate))

prediction = model.predict(bar1[500:502,:,:])
print("Prediction: ", prediction)
print("Actual: ", bar2[500:502,:,:])
