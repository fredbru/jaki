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
    allhits.append(hits)
    alltiming.append(timing)
    alltempo.append(tempo)
    print(name, len(hits))
    grooveCount += len(hits)

np.save(allhits, "Hits.npy")
np.save(alltiming, "Timing.npy")
np.save(alltempo, "Tempo.npy")