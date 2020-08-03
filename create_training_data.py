# Create training dataset from BFD or MIDI files. Uses GrooveToolbox.
import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd


sys.path.append("/home/fred/BFD/python/GrooveToolbox/")
from Groove import *
from LoadGrooveFromBFDPalette import *

directory = "DATASET/"
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
    groove5Parts[groove5Parts < 0.3] = 0
    return groove5Parts

def encodeCategorical(allGrooves):
    # Encode BFD format grooves into
    # 0 - Kick
    # 1 - Snare
    # 2 - Closed cymbals (hihat and ride)
    # 3 - Open cymbals (open hihat, crash and extra cymbal
    # 4 - Toms (low mid and high)

    categories = ['ko', 'r', 'c', 's', 'k', 't', 'kc', 'kot', 'o', 'sc', 'kcot',
                  'ct', 'so', 'kco', 'co', 'kso', 'ks', 'kt', 'st', 'ksc', 'ksco',
                  'sco', 'ksct', 'kct', 'kst', 'sct', 'ot', 'cot', 'ksot', 'kscot',
                  'sot', 'scot']

    allCategorical = np.empty([allGrooves.shape[0],32], dtype='str')
    oneHotPatterns = np.empty([allGrooves.shape[0],32,32])

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
        oh = oh.reindex(sorted(oh.columns), axis=1)
        oneHotPatterns[j,:,:] = oh.to_numpy()
    np.random.shuffle(oneHotPatterns)
    return oneHotPatterns

for i in range(len(folders)):
    bundle = np.load(directory + "/" + folders[i] + "/Hits.npy")
    #print(folders[i])
    shortBundle = np.zeros([bundle.shape[0],32,5])
    for j in range(bundle.shape[0]):
        shortBundle[j,:,:] = np.ceil(_groupGroove5KitParts(bundle[j,:,:]))
    allGrooves = np.vstack((allGrooves,shortBundle))

rowsToDelete = [] #remove any weird loops with too many rests
for i in range(allGrooves.shape[0]):
    bar2 = allGrooves[i, 16:32, :]
    if np.count_nonzero(bar2) < 2:
        rowsToDelete.append(i)
allGrooves = np.delete(allGrooves, rowsToDelete , axis=0)
print(allGrooves.shape)


oneHotPatterns = encodeCategorical(allGrooves)
np.save("One-Hot-Loops-3-7.npy", oneHotPatterns)
