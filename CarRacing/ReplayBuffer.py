import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, maxSize, inputShape, nActions, discrete=False, stateType=np.float32): 
        self.memSize = maxSize
        self.discrete = discrete

        # Handle case when state inputShape is multiDimensional
        if (isinstance(inputShape, int)):   
            self.stateMemory = np.zeros((self.memSize, inputShape), dtype=stateType)
            self.newStateMemory = np.zeros((self.memSize, inputShape), stateType)
        else:
            self.stateMemory = np.zeros((self.memSize, *inputShape), dtype=stateType)
            self.newStateMemory = np.zeros((self.memSize, *inputShape), dtype=stateType)


        # discrete vs continuous action space
        _dtype = np.int8 if self.discrete else np.float32
        self.actionMemory = np.zeros((self.memSize, nActions), dtype=_dtype)


        self.rewardMemory = np.zeros(self.memSize, dtype=np.float32)
        self.terminalMemory = np.zeros(self.memSize, dtype=np.int8)

        self.memPtr = 0
        self.rollOver = 0
    

    def store_transition(self, state, action, reward, newState, done):
        # If at end of array roll pointer back to start
        if (self.memPtr >= self.memSize):
            self.memPtr = 0
            self.rollOver = 1
            print("**Replay buffer pointer rollover")


        self.stateMemory[self.memPtr] = state
        self.newStateMemory[self.memPtr] = newState

        self.rewardMemory[self.memPtr] = reward

        # Should be false when episode is over, and true when episode is still going 
        self.terminalMemory[self.memPtr] = 1 - int(done)

        # If discrete we one hot encode
        if self.discrete:
            actions = np.zeros(self.actionMemory.shape[1])
            actions[action] = 1.0
            self.actionMemory[self.memPtr] = actions
        else:
            self.actionMemory[self.memPtr] = action

        self.memPtr += 1


    def sample_buffer(self, batchSize):
        # If a rollover has not happened yet get batch from available samples
        if (self.rollOver == 0):
            maxMem = min(self.memPtr, self.memSize)
            batch = np.random.choice(maxMem, batchSize)

        # If a rollover has happened, use the entire buffer
        else:
            batch = np.random.choice(self.memSize, batchSize)

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

        with open(str(fileName) + "_pointer" + '.pkl', 'wb') as file:
            pickle.dump([self.memPtr, self.rollOver], file)


    def load_buffer(self, fileName):
        self.clearBuffer()

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
        
        with open(str(fileName) + "_pointer" + '.pkl', 'rb') as file:
            temp = pickle.load(file)
            self.memPtr = temp[0]
            self.rollOver = temp[1]


    def clearBuffer(self):
        del self.stateMemory
        del self.newStateMemory
        del self.rewardMemory
        del self.actionMemory
        del self.terminalMemory