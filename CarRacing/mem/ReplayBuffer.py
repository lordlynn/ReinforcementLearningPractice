import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, maxSize, inputShape, nActions, discrete=False): 
        # TODO : USe the most efficient data types
        
        self.memSize = maxSize
        self.discrete = discrete

        # What is the use of this?? discrete vs continuous
        _dtype = np.int8 if self.discrete else np.float32

        # Handle case when inputShape is multiDimensional
        if (isinstance(inputShape, int)):   
            self.stateMemory = np.zeros((self.memSize, inputShape))
            self.newStateMemory = np.zeros((self.memSize, inputShape))
        else:
            self.stateMemory = np.zeros((self.memSize, *inputShape))
            self.newStateMemory = np.zeros((self.memSize, *inputShape))

        self.actionMemory = np.zeros((self.memSize, nActions), dtype=_dtype)
        self.rewardMemory = np.zeros(self.memSize)
        self.terminalMemory = np.zeros(self.memSize, dtype=np.float32)

        self.memPtr = 0
    

    def store_transition(self, state, action, reward, newState, done):
        # First available memory
        if (self.memPtr >= self.memSize):
            self.memPtr = 0
            print("**Replay buffer pointer rollover")

        self.stateMemory[self.memPtr ] = state
        self.newStateMemory[self.memPtr ] = newState

        self.rewardMemory[self.memPtr ] = reward

        # Should be false when episode is over, and true when episode is still going 
        self.terminalMemory[self.memPtr ] = 1 - int(done)

        # If discrete we one hot encode
        if self.discrete:
            actions = np.zeros(self.actionMemory.shape[1])
            actions[action] = 1.0
            self.actionMemory[self.memPtr ] = actions
        else:
            self.actionMemory[self.memPtr ] = action

        self.memPtr += 1


    def sample_buffer(self, batchSize):
        maxMem = min(self.memPtr, self.memSize)
        batch = np.random.choice(maxMem, batchSize)

        states = self.stateMemory[batch]
        newStates = self.newStateMemory[batch]
        rewards = self.rewardMemory[batch]
        actions = self.actionMemory[batch]
        terminal = self.terminalMemory[batch]

        return states, actions, rewards, newStates, terminal 
        
    def save_buffer(self, fileName):
        with open(str(fileName) + "_states" + '.pkl', 'wb') as file:
            pickle.dump(self.stateMemory, file)

        with open(str(fileName) + "_newStates" + '.pkl', 'wb') as file:
            pickle.dump(self.newStateMemory, file)

        with open(str(fileName) + "_reward" + '.pkl', 'wb') as file:
            pickle.dump(self.rewardMemory, file)

        with open(str(fileName) + "_action" + '.pkl', 'wb') as file:
            pickle.dump(self.actionMemory, file)

        with open(str(fileName) + "_terminal" + '.pkl', 'wb') as file:
            pickle.dump(self.terminalMemory, file)

    def load_buffer(self, fileName):
        self.clearBuffer()
        sizes = []

        with open(str(fileName) + "_states" + '.pkl', 'rb') as file:
            self.stateMemory = pickle.load(file)
    
        with open(str(fileName) + "_newStates" + '.pkl', 'rb') as file:
            self.newStateMemory = pickle.load(file)

        with open(str(fileName) + "_reward" + '.pkl', 'rb') as file:
            self.rewardMemory = pickle.load(file)

        with open(str(fileName) + "_action" + '.pkl', 'rb') as file:
            self.actionMemory = pickle.load(file)

        with open(str(fileName) + "_terminal" + '.pkl', 'rb') as file:
            self.terminalMemory = pickle.load(file)

        sizes.append(self.stateMemory.shape[0])
        sizes.append(self.newStateMemory.shape[0])
        sizes.append(self.rewardMemory.shape[0])
        sizes.append(self.actionMemory.shape[0])
        sizes.append(self.terminalMemory.shape[0]) 

        if (len(set(sizes)) == 1):
             self.memPtr = sizes[0]

    def clearBuffer(self):
        del self.stateMemory
        del self.newStateMemory
        del self.rewardMemory
        del self.actionMemory
        del self.terminalMemory