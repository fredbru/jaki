import numpy as np
np.set_printoptions(threshold=np.inf,precision=2)

def act(state):
    done = False
    actionArray, action = getRandomAction()  # pick random action
    print(action)
    print(actionArray)
    print(np.reshape(action, (16,32)))
    print(np.array_equal(np.reshape(action, (16,32)),actionArray))
    # here need to carry out action
    actionIndex = np.nonzero(actionArray)
    new_state = np.copy(state)
    new_state[actionIndex[0], :] = 0
    new_state[actionIndex] = 1
    reward = calculateSyncopation(new_state) - calculateSyncopation(state)  # reward should be difference in syncopation
    if reward > 8.0:
        done = True
    return new_state, reward, done, action


def getRandomAction():
    # make an array with a random 1 in it. this 1 corresponds to a change in a single value for any position in the
    # loop state (for example setting a rest into a snare hit at position 5 etc).
    actionArray = np.zeros([16, 32])
    actionIndex = np.random.randint(16), np.random.randint(32)
    actionArray[actionIndex] = 1
    action = np.reshape(actionArray, (512))
    return actionArray, action


def calculateSyncopation(state):
    # calculate syncopation reward. should this be offset against original syncpation? so
    # rewarding increase in syncopation
    # test this against something
    # theoretical max is like 15 or something.
    # If this is too slow for training, could be a way to optimize?
    metricalProfile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
    syncopation = 0.0
    state[:,22] = 0.0 # set rests in one-hot vector to 0.
    for i in range(16):
        if 1 in state[i, :]:
            if np.sum(state[(i + 1) % 16, :]) == 0.0 and metricalProfile[(i + 1) % 16] > metricalProfile[i]:
                syncopation = float(syncopation + (
                    abs(metricalProfile[(i + 1) % 16] - metricalProfile[i])))

            elif np.sum(state[(i + 2) % 16, :]) == 0.0 and metricalProfile[(i + 2) % 16] > metricalProfile[i]:
                syncopation = float(syncopation + (
                    abs(metricalProfile[(i + 2) % 16] - metricalProfile[i])))
    return syncopation

def convertOneHotToList(groove):
    # convert one hot or probabilities matrix from LSTM model into a list format
    columns = ['c ', 'co', 'cot', 'ct', 'k ', 'kc', 'kco', 'kcot', 'kct', 'ko', 'kot',
               'ks', 'ksc', 'ksco', 'kscot', 'ksct', 'kso', 'ksot', 'kst', 'kt', 'o ',
               'ot', '  ', 's ', 'sc', 'sco', 'scot', 'sct', 'so', 'sot', 'st', 't ']

    grooveList = []
    for i in range(groove.shape[0]):
        index = np.argmax(groove[i,:])
        grooveList.append(columns[index])
    return grooveList

oneHotGrooves = np.load("One-Hot-Grooves-Nonswung.npy")

bar1 = oneHotGrooves[6000,0:16,:]
bar2 = oneHotGrooves[6000,16:32,:]


for i in range(1):
    act(oneHotGrooves[i,0:16,:])
