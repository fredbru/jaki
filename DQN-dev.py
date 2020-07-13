import numpy as np
import random
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random
import time

from collections import deque

np.set_printoptions(threshold=np.inf,precision=2)

class DQN:
    def __init__(self, LSTM, seedLoop):
        # Initialize Deep-Q reinforcement learning model. Consists of 2 networks of same architecture- Q-network to
        # choose next action and Target-Q to estimate expected future return.

        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.999
        self.learningRate = 0.001
        self.tau = .125
        self.seedLoop = seedLoop

        self.probabilities = LSTM.predict(np.expand_dims(seedLoop, 0))[0]
        print("LSTM predictions ", convertOneHotToList(self.probabilities))
        print("LSTM prediction = ", convertOneHotToList(self.probabilities))
        self.mostLikely = np.zeros([16,32])
        for i in range(self.probabilities.shape[0]):
            index = np.argmax(self.probabilities[i,:])
            self.mostLikely[i,index] = 1.0

        self.model = self.create_model()
        self.targetModel = self.create_model()
        self.calculateKickDensity(seedLoop)
        self.calculateSymmetry(seedLoop)

    def create_model(self):
        model = Sequential()
        #state_shape = [16,32]
        model.add(Dense(16, input_shape=(16,32)))
        model.add(Dense(24, input_shape=(16,32), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(32))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learningRate))
        model.summary()
        weights = model.get_weights()
        weights[0] = self.probabilities.T
        model.set_weights(weights)
        return model

    def act(self, state):
        self.epsilon *= self.epsilonDecay
        self.epsilon = max(self.epsilonMin, self.epsilon)
        done = False
        if np.random.random() < self.epsilon:
            # actions are adding or removing any number of onsets
            action, actionIndex = self.getRandomAction()  # pick random action
        else:
            prediction = self.model.predict(state) #output shape 16,32
            actionIndex = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape) # do predicted best action
            actionIndex = actionIndex[1], actionIndex[2]
            action = np.zeros([1,16,32])
            action[0,actionIndex[0],actionIndex[1]] = 1

        # Carry out action. problem - model output is 512 1D vector.
        newState = np.copy(state)
        newState[0,actionIndex[0],:] = 0
        newState[0,actionIndex[0],actionIndex[1]] = 1

        reward = self.getReward(state,newState)

        if reward > 5.0:
            done = True
        return newState, reward, done, action

    def getReward(self, state, newState):
        # Calculate reward as sum of complexity increase (syncopation+density) and similarity to probabilities.
        # todo: optimize weights, find a better way to combine features.

        syncopationReward = self.calculateCombinedMonoSyncopation(newState[0,:,:]) -\
                            self.calculateCombinedMonoSyncopation(self.seedLoop)
        distancePenalty = np.sum(np.abs(newState[0,:,:] - self.mostLikely))/2.0
        #print("jaki ", convertOneHotToList(newState[0,:,:]))
        print("seed ", convertOneHotToList(self.seedLoop))
        #print("lstm ", convertOneHotToList(self.mostLikely))
        densityDifference = self.calculateOverallDensity(newState[0,:,:]) - self.calculateOverallDensity(self.seedLoop)
        densityReward =  -(densityDifference - 2) # reward for being close to 2
        # todo: change density reward to fixed number - not difference?

        reward = syncopationReward - distancePenalty + densityReward
        #print(syncopationReward, -distancePenalty, densityReward)
        return reward

    def getRandomAction(self):
        # make an array with a random 1 in it. this 1 corresponds to a change in a single value for any position in the
        # loop state (for example setting a rest into a snare hit at position 5 etc).
        # change actions to prioritize removing onsets more.
        actionArray = np.zeros([16, 32])

        # exclude physically impossible hit combinations and rests
        possibleHits = [0,3,4,5,8,9,10,11,12,15,18,19,20,21,23,24,27,28,30,31]

        # 50/50 chance of either make a rest or change a note
        if np.random.randint(0,10) == 1:
            actionIndex = np.random.randint(16), random.choice(possibleHits)
        else:
            actionIndex = np.random.randint(16), 22
        actionArray[actionIndex[0],actionIndex[1]] = 1
        actionIndex = np.nonzero(actionArray)

        return actionArray, actionIndex

    def calculateOverallDensity(self, loop):
        # calculate density, counting indexes of multiple coincident onsets and not counting rests
        restIndex = [22]
        density = 0
        singleHitIndexes = [0,4,20,23,31]
        doubleHitIndexes = [1,3,5,9,11,19,21,24,28,30]
        tripleHitIndexes = [2,6,8,10,12,16,18,25,27,29]
        quadrupleHitIndexes = [7,13,15,17,26]
        quintupleHitIndex = [14]

        for i in range(16):
            hit = np.nonzero(loop[i,:])[0]

            if hit in singleHitIndexes:
                density+=1
            if hit in doubleHitIndexes:
                density+=2
            if hit in tripleHitIndexes:
                density+=3
            if hit in quadrupleHitIndexes:
                density+=4
            if hit in quintupleHitIndex:
                density+=5
        end = time.time()
        return density

    def calculateKickDensity(self, loop):
        # Count number of kicks vs length of bar
        #s = time.time()
        kickDensity = np.sum([loop[:,4],loop[:,5],loop[:,6],loop[:,7],loop[:,8],
                              loop[:, 9],loop[:,10],loop[:,11],loop[:,12],loop[:,13],
                              loop[:, 14],loop[:,15],loop[:,16],loop[:,17],loop[:,18],
                              loop[:,19]]) / 16.0
        #e = time.time()
        #print(e-s)
        return kickDensity

    def calculateSnareDensity(self, loop):
        # Count number of snares vs length of bar
        snareDensity = np.sum([loop[:,11],loop[:,12],loop[:,13],loop[:,14],loop[:,15],
                              loop[:, 16],loop[:,17],loop[:,18],loop[:,23],loop[:,24],
                              loop[:, 25],loop[:,26],loop[:,27],loop[:,28],loop[:,29],
                              loop[:,30]]) / 16.0
        return snareDensity

    def calculateCymbalDensity(self, loop):
        # Count number of cymbal hits (closed or open) in loop vs length of bar
        cymbalDensity = np.sum([loop[:,0],loop[:,1],loop[:,2],loop[:,3],loop[:,5],
                              loop[:, 6],loop[:,7],loop[:,8],loop[:,9],loop[:,10],
                              loop[:, 12],loop[:,13],loop[:,14],loop[:,15],loop[:,16],
                              loop[:,17], loop[:, 20], loop[:,21], loop[:, 24], loop[:,25],
                              loop[:,27],loop[:,28],loop[:,29]]) / 16.0
        return cymbalDensity

    def calculateTomDensity(self, loop):
        tomDensity = np.sum([loop[:,2],loop[:,3],loop[:,7],loop[:,8],loop[:,10],
                              loop[:,14],loop[:,15],loop[:,17],loop[:,18],loop[:,19],
                              loop[:,21],loop[:,26],loop[:,27],loop[:,29],loop[:,30],
                              loop[:,31]]) / 16.0
        return tomDensity

    def calculateCombinedMonoSyncopation(self, state):
        # calculate syncopation for 1 bar loop using Longuet-Higgins monophonic syncopation model
        metricalProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
        syncopation = 0.0
        loop = np.copy(state)
        loop[:, 22] = 0.0  # set rests in one-hot vector to 0.
        for i in range(16):
            if 1 in loop[i, :]:
                if np.sum(loop[(i + 1) % 16, :]) == 0.0 and metricalProfile[(i + 1) % 16] > metricalProfile[i]:
                    syncopation = float(syncopation + (
                        abs(metricalProfile[(i + 1) % 16] - metricalProfile[i])))

                elif np.sum(loop[(i + 2) % 16, :]) == 0.0 and metricalProfile[(i + 2) % 16] > metricalProfile[i]:
                    syncopation = float(syncopation + (
                        abs(metricalProfile[(i + 2) % 16] - metricalProfile[i])))
        return syncopation

    def calculateSymmetry(self, loop):
        # Calculate symmetry for any number of parts.
        # Defined as the the number of positions in the first and second halves that have the same value (rest or hit)
        # divided by the total number of onsets in the pattern. As perfectly symmetrical pattern
        # would have a symmetry of 1.0
        # Doesn't deal with simultaneous onsets properly - still considers them different events
        # NB unlike groove toolbox, counts rests occuring at the same place too.

        part1,part2 = np.split(loop,2, axis=0)
        index1 = np.nonzero(part1.flatten())
        index2 = np.nonzero(part2.flatten())
        symmetry = np.intersect1d(index1, index2).size * 2.0 / np.count_nonzero(loop)

        return symmetry

    def remember(self, currentState, action, reward, newState, done):
        self.memory.append([currentState, action, reward, newState, done])

    def replay(self):
        # Train Q-network
        batchSize = 1024
        if len(self.memory) < batchSize:
            return

        samples = random.sample(self.memory, batchSize)

        currentStates = np.zeros([batchSize,16,32])
        actions = np.zeros([batchSize,16,32])
        rewards = np.zeros([batchSize,1])
        newStates = np.zeros([batchSize,16,32])
        dones = np.zeros([batchSize,1])

        for i in range(batchSize):
            sample = samples[i]
            currentState, action, reward, newState, done = sample
            currentStates[i,:,:] = currentState
            actions[i,:,:] = action
            rewards[i,0] = reward
            newStates[i,:,:] = newState
            dones[i,0] = done


        start = time.time()
        # todo: optimize? this is the problem bit
        target = self.targetModel.predict(currentStates) #target is the q value?

        Q_future = np.amax(self.targetModel.predict(newStates,batch_size=batchSize), axis=(1,2)).reshape(batchSize,1,1)

        # actions array = 1 for action, 0 everywhere else. so sets all non action values to 0
        target = rewards.reshape(batchSize,1,1) + Q_future * self.gamma * actions
        #target[actionIndex] = rewards + Q_future * self.gamma

        self.model.fit(currentStates, target, epochs=1, verbose=0)
        # doneIndexes = np.where(dones)
        # print(doneIndexes)
        # target = target * rewards
        end = time.time()
        #print("Batch training time = ", end-start)

    def target_train(self):
        # Train target-Q network
        # todo: optimize this?
        weights = self.model.get_weights()
        target_weights = self.targetModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.targetModel.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def initWeights(self,shape):
        return self.probabilities

def convertOneHotToList(groove):
    # convert one hot or probabilities matrix from LSTM model into a list format for quick reading
    columns = [' c ', 'co ', 'cot', 'ct ', ' k ', 'kc ', 'kco', 'kcot', 'kct', 'ko ', 'kot',
               'ks ', 'ksc', 'ksco', 'kscot', 'ksct', 'kso', 'ksot', 'kst', 'kt ', ' o ',
               'ot ', '---', ' s ', 'sc ', 'sco', 'scot', 'sct', 'so ', 'sot', 'st ', ' t ']
    grooveList = []
    for i in range(groove.shape[0]):
        index = np.argmax(groove[i, :])
        grooveList.append(columns[index])
    return grooveList

oneHotGrooves = np.load("One-Hot-Drum-Loops.npy")
oneBarGrooves = oneHotGrooves[:,0:16,:]

LSTM = load_model("JAKI_Encoder_Decoder_29-6_4")

gamma = 0.9
epsilon = .95

trials = 1000
trial_len = 1500

steps = []
for trial in range(trials):
    # reset environment to a random loop
    seedLoop = oneBarGrooves[np.random.randint(0,6244),:,:]
    dqnAgent = DQN(LSTM, seedLoop)

    currentState = np.copy(seedLoop).reshape(1,seedLoop.shape[0],seedLoop.shape[1])

    for step in range(trial_len):
        newState, reward, done, action = dqnAgent.act(currentState) #todo: do the action
        # action shape = 16,32 size
        dqnAgent.remember(currentState, action, reward, newState, done)

        dqnAgent.replay()  # internally iterates default (prediction) model
        dqnAgent.target_train()  # iterates target model

        currentState = np.copy(newState)
        if done:
            print("Seed Loop", convertOneHotToList(seedLoop))
            print("LSTM Loop", convertOneHotToList(LSTM.predict(np.expand_dims(seedLoop, 0))[0]))
            print("DQN  Loop", convertOneHotToList(newState[0,:,:]))
            break
    if step >= 500:
        print("Failed to complete in trial {}".format(trial))
        if step % 10 == 0:
            dqnAgent.save_model("trial-{}.model".format(trial))
    else:
        print("Completed in {} trials".format(trial))
        dqnAgent.save_model("success.model")
        break
