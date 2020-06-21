import numpy as np
import random
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random

from collections import deque

np.set_printoptions(threshold=np.inf,precision=2)

class DQN:
    def __init__(self, LSTM, seedLoop):
        # Initialize Deep-Q reinforcement learning model. Consists of 2 networks of same architecture- Q-network to
        # choose next action and Target-Q to estimate expected future return.

        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
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
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        #state_shape = [16,32]
        model.add(Dense(16, input_shape=(32,)))
        model.add(Dense(24, input_shape=(16,32), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(32))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        weights = model.get_weights()
        weights[0] = self.probabilities.T
        model.set_weights(weights)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        done = False
        if np.random.random() < self.epsilon:
            # actions are adding or removing any number of onsets
            action, actionIndex = self.getRandomAction()  # pick random action
        else:
            prediction = self.model.predict(state)
            actionIndex = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape) # do predicted best action
            action = np.zeros([16,32])
            action[actionIndex[0],actionIndex[1]] = 1

        # Carry out action. problem - model output is 512 1D vector.
        newState = np.copy(state)
        newState[actionIndex[0],:] = 0
        newState[actionIndex[0],actionIndex[1]] = 1

        reward = self.getReward(state,newState)

        if reward > 5.0:
            done = True
        return newState, reward, done, action

    def getReward(self, state, newState):
        # Calculate reward as sum of complexity increase (syncopation+density) and similarity to probabilities.
        # todo: optimize weights, find a better way to combine features.

        syncopationReward = self.calculateSyncopation(newState) - self.calculateSyncopation(self.seedLoop)
        distancePenalty = np.sum(np.abs(newState - self.mostLikely))/2.0
        # print("jaki ", convertOneHotToList(newState))
        # print("seed ", convertOneHotToList(self.seedLoop))
        # print("lstm ", convertOneHotToList(self.mostLikely))
        densityDifference = self.calculateDensity(newState) - self.calculateDensity(self.seedLoop)
        densityReward =  -(densityDifference - 5) # reward for being close to 5

        reward = syncopationReward - distancePenalty + densityReward
        print(syncopationReward, -distancePenalty, densityReward)
        return reward

    def getRandomAction(self):
        # make an array with a random 1 in it. this 1 corresponds to a change in a single value for any position in the
        # loop state (for example setting a rest into a snare hit at position 5 etc).
        # change actions to prioritize removing onsets more.
        actionArray = np.zeros([16, 32])

        # exclude physically impossible hit combinations and rests
        possibleHits = [0,3,4,5,8,9,10,11,12,15,18,19,20,21,23,24,27,28,30,31]

        # 50/50 chance of either make a rest or change a note
        if np.random.randint(0,2) == 1:
            actionIndex = np.random.randint(16), random.choice(possibleHits)
        else:
            actionIndex = np.random.randint(16), 22
        actionArray[actionIndex[0],actionIndex[1]] = 1
        actionIndex = np.nonzero(actionArray)

        return actionArray, actionIndex

    def calculateDensity(self, loop):
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
        return density / 2.0

    def calculateSyncopation(self, state):
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

    def remember(self, currentState, action, reward, newState, done):
        self.memory.append([currentState, action, reward, newState, done])

    def replay(self):
        # Train Q-network
        batch_size = 1024
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            currentState, action, reward, newState, done = sample
            target = self.target_model.predict(currentState)
            if done:
                target[0] = reward
            else:
                Q_future = np.max(self.target_model.predict(newState))
                target[0] = reward + Q_future * self.gamma

            self.model.fit(currentState, target, epochs=1, verbose=0)

    def target_train(self):
        # Train target-Q network
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

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

LSTM = load_model("JAKI_Encoder_Decoder_14-6-20_2")

gamma = 0.9
epsilon = .95

trials = 1000
trial_len = 1500


steps = []
for trial in range(trials):
    # reset environment to a random loop
    seedLoop = oneBarGrooves[np.random.randint(0,6244),:,:]
    dqn_agent = DQN(LSTM, seedLoop)

    currentState = np.copy(seedLoop)

    for step in range(trial_len):
        newState, reward, done, action = dqn_agent.act(currentState) #todo: do the action
        dqn_agent.remember(currentState, action, reward, newState, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        currentState = np.copy(newState)
        if done:
            print("Seed Loop", convertOneHotToList(seedLoop))
            print("LSTM Loop", convertOneHotToList(LSTM.predict(np.expand_dims(seedLoop, 0))[0]))
            print("DQN  Loop", convertOneHotToList(currentState))
            break
    if step >= 199:
        print("Failed to complete in trial {}".format(trial))
        if step % 10 == 0:
            dqn_agent.save_model("trial-{}.model".format(trial))
    else:
        print("Completed in {} trials".format(trial))
        dqn_agent.save_model("success.model")
        break
