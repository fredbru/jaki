import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from collections import deque


np.set_printoptions(threshold=np.inf,precision=2)

sys.path.append("/home/fred/BFD/python/GrooveToolbox/")
from Groove import *
from LoadGrooveFromBFDPalette import *

model = load_model('JAKI_Encoder_Decoder_12-6-20')
model.summary()

oneHotGrooves = np.load("One-Hot-Grooves-Nonswung.npy")
bar1 = oneHotGrooves[6000:,0:16,:]
bar2 = oneHotGrooves[6000:,16:32,:]

firstBar = bar1[0:100,:,:]
prediction = model.predict(bar1[0:100,:,:])
print(prediction.shape, 'shape')
target = bar2[0:100,:,:]

columns = ['c ', 'co', 'cot', 'ct', 'k ', 'kc', 'kco', 'kcot', 'kct', 'ko', 'kot',
 'ks', 'ksc', 'ksco', 'kscot', 'ksct', 'kso', 'ksot', 'kst', 'kt', 'o ',
 'ot',  '  ',  's ',  'sc',  'sco',  'scot',  'sct',  'so',  'sot',  'st',  't ']

def convertOneHotToList(groove):
    # convert one hot or probabilities matrix from LSTM model into a list format
    grooveList = []
    for i in range(groove.shape[0]):
        index = np.argmax(groove[i,:])
        grooveList.append(columns[index])
    return grooveList

def convertListToMatrix(grooveList):
    # convert list format groove into matrix for using groovetoolbox functions.
    pass


num_actions = 20

eg = model.predict(bar1[0:2,:,:])

#build rl model

obsspace = eg
actspace = eg

print(convertOneHotToList(eg))




# n = 0
# for n in range(100):
#     print("Ref ---", '  1     2    3     4    5    6     7    8    9    10    11    12   13    14    15    16')
#     print('1st Bar' + str(convertOneHotToList(firstBar[n,:,:])))
#     #print('2nd Bar' + str(convertOneHotToList(target[n,:,:])))
#     print('Gen Bar' + str(convertOneHotToList(prediction[n,:,:])))
#     print("\n")

