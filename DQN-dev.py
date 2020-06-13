import numpy as np
import random
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

np.set_printoptions(threshold=np.inf,precision=2)

class DQN:
    def __init__(self, LSTM, seedLoop):
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.probabilities = LSTM.predict(np.expand_dims(seedLoop, 0))[0]
        print("LSTM prediction = ", convertOneHotToList(self.probabilities))
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        #state_shape = [16,32]
        model.add(Dense(16, input_shape=(32,)))
        model.add(Dense(24, input_shape=(16,32), activation="relu")) # does this need to be 512? size of state space? flatten model?
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(32)) #number of possible actions
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
            # todo: action space is the set of all possible actions
            # actions are adding or removing any number of onsets
            action, actionIndex = self.getRandomAction()  # pick random action
        else:
            prediction = self.model.predict(state)
            actionIndex = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape) # do predicted best action. add batch size
            action = np.zeros([16,32])
            action[actionIndex] = 1
            #actionArray = np.reshape(action, (16, 32))

        # Carry out action. problem - model output is 512 1D vector.
        newState = np.copy(state)
        newState[actionIndex[0],:] = 0
        newState[actionIndex] = 1
        reward = self.calculateSyncopation(newState) - self.calculateSyncopation(state)  # reward is difference in syncopation
        if reward > 3.0:
            done = True
            print(reward)
        return newState, reward, done, action

    def getRandomAction(self):
        # make an array with a random 1 in it. this 1 corresponds to a change in a single value \for any position in the
        # loop state (for example setting a rest into a snare hit at position 5 etc).
        actionArray = np.zeros([16, 32])
        actionIndex = np.random.randint(16), np.random.randint(32)
        actionArray[actionIndex] = 1
        action = np.reshape(actionArray, (512))
        actionIndex = np.nonzero(actionArray)
        return actionArray, actionIndex

    def calculateSyncopation(self, state):
        # calculate syncopation reward. should this be offset against original syncpation? so
        # rewarding increase in syncopation
        # test this against something
        # theoretical max is like 15 or something.
        # If this is too slow for training, could be a way to optimize?
        metricalProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
        syncopation = 0.0
        state[:, 22] = 0.0  # set rests in one-hot vector to 0.
        for i in range(16):
            if 1 in state[i, :]:
                if np.sum(state[(i + 1) % 16, :]) == 0.0 and metricalProfile[(i + 1) % 16] > metricalProfile[i]:
                    syncopation = float(syncopation + (
                        abs(metricalProfile[(i + 1) % 16] - metricalProfile[i])))

                elif np.sum(state[(i + 2) % 16, :]) == 0.0 and metricalProfile[(i + 2) % 16] > metricalProfile[i]:
                    syncopation = float(syncopation + (
                        abs(metricalProfile[(i + 2) % 16] - metricalProfile[i])))
        return syncopation

    def remember(self, currentState, action, reward, newState, done):
        self.memory.append([currentState, action, reward, newState, done])

    def replay(self): # need to refit this function to use new adapted reward calculation?
        # this is not training the target -  this is training self.model, using target estimated by
        # by target model
        batch_size = 1024
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            currentState, action, reward, newState, done = sample
            target = self.target_model.predict(currentState)
            # print(target[0].shape, 'shape') # target is an action. target[0] = 16x32
            if done:
                target[0] = reward
            else:
                Q_future = np.max(self.target_model.predict(newState))
                # print(Q_future)
                target[0] = reward + Q_future * self.gamma
            # print(target.shape)
            self.model.fit(currentState, target, epochs=1, verbose=0) #fit updates model weights

    def target_train(self):
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
    # convert one hot or probabilities matrix from LSTM model into a list format
    columns = ['c ', 'co', 'cot', 'ct', 'k ', 'kc', 'kco', 'kcot', 'kct', 'ko', 'kot',
               'ks', 'ksc', 'ksco', 'kscot', 'ksct', 'kso', 'ksot', 'kst', 'kt', 'o ',
               'ot', '  ', 's ', 'sc', 'sco', 'scot', 'sct', 'so', 'sot', 'st', 't ']
    grooveList = []
    for i in range(groove.shape[0]):
        index = np.argmax(groove[i, :])
        grooveList.append(columns[index])
    return grooveList


gamma = 0.9
epsilon = .95

trials = 1000
trial_len = 1500

oneHotGrooves = np.load("One-Hot-Grooves-Nonswung.npy")
oneBarGrooves = oneHotGrooves[:,0:16,:]

LSTM = load_model("JAKI_Encoder_Decoder_12-6-20")


# updateTargetNetwork = 1000

steps = []
for trial in range(trials):
    # reset environment to a random loop
    seedLoop = oneBarGrooves[np.random.randint(0,6000),:,:]
    dqn_agent = DQN(LSTM, seedLoop)

    # print(LSTM.summary())
    # LSTMProbability = LSTM.predict(np.expand_dims(groove, 0))[0]
    # print('\n')
    # print(dqn_agent.model.layers[0].get_weights()[1].shape)
    # dqn_agent.model.layers[0].set_weights(LSTMProbability)
    currentState = seedLoop

    for step in range(trial_len):
        newState, reward, done, action = dqn_agent.act(currentState) #todo: do the action
        dqn_agent.remember(currentState, action, reward, newState, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        currentState = newState
        if done:
            print("1st", convertOneHotToList(seedLoop))
            print("Gen", convertOneHotToList(currentState))
            break
    if step >= 1499:
        print("Failed to complete in trial {}".format(trial))
        if step % 10 == 0:
            dqn_agent.save_model("trial-{}.model".format(trial))
    else:
        print("Completed in {} trials".format(trial))
        dqn_agent.save_model("success.model")
        break
