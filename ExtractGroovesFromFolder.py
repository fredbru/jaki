import sys
import os
import numpy as np
sys.path.append("/home/fred/BFD/python/GrooveToolbox/")
from Groove import *
from LoadGrooveFromBFDPalette import *

files = os.listdir("/home/fred/BFD/python/jaki/Grooves/")
allhits = []
alltiming = []
alltempo = []
grooveCount = 0

for name in files:
    hits, timing, tempo = getAllGroovesFromBFDPalette(name)
    print(len(hits))
    print(hits[1].shape)
    filename = name[:-8]
    print(filename)
    Folder = filename
    os.mkdir(Folder)
    np.save(Folder + "/" + "Hits.npy", hits)
    np.save(Folder + "/" + "Timing.npy", timing)
    np.save(Folder + "/" + "Tempo.npy", tempo)

    print(name, len(hits))
    grooveCount += len(hits)

