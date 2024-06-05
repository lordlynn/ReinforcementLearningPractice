import numpy as np

class ReplayBuffer(object):
    def __init__(self, maxSize, inputShape, nActions, discrete=False):
        self.memSize = maxSize
        self.discrete = discrete

        # What is the use of this?? discrete vs continuous
        dtype = np.int8 if self.discrete else np.float32

        self.stateMemory = np.zeros((self.memSize, inputShape))
        self.newStateMemory = np.zeros((self.memSize, inputShape))

        self.actionMemory = np.zeros((self.memSize, nActions))
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

