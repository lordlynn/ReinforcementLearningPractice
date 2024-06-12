import keras
from keras import layers
import keras.optimizers
import numpy as np
import pickle

import ReplayBuffer

class Agent(object):
    def __init__(self, learningRate=0.001, nActions=None, gamma=0.99, batchSize=None, inputDims=None, epsilon=None, epsilonDec=0.996, epsilonEnd=0.01, memSize=100000, running=False):
        # If training is being performed
        if (running == False):
            check = [nActions, batchSize, inputDims, epsilon]

            if (None in check):
                raise ValueError("Make sure when training every argument is set")

            self.actionsSpace = [i for i in range(nActions)]
            
            self.nActions = nActions                            # Number of actions
            self.inputDims = inputDims                          # Number of inputs (state)
            self.learningRate = learningRate                    # Learning Rate
            self.gamma = gamma                                  # Discount Factor
            self.epsilon = epsilon                              # Exploration rate
            self.epsilonDec = epsilonDec                        # Decay rate of epsilon
            self.epsilonMin = epsilonEnd                        # Minimum value of epsilon
            self.batchSize = batchSize                          # Training batch size

            self.memory = ReplayBuffer.ReplayBuffer(memSize, self.inputDims[:-1], nActions, discrete=True, stateType=np.uint8)
        else:
            self.learningRate = learningRate  
            self.nActions = nActions
            self.epsilon = 0                                   
            self.epsilonDec = 0                        
            self.epsilonMin = 0                        

    def build_network(self):
        if (isinstance(self.inputDims, int)):
            inputs = layers.Input(shape=(self.inputDims,))
        else:
            inputs = layers.Input(shape=self.inputDims)


        HL1 = layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu') (inputs)

        HL2 = layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu') (HL1)

        HL3 = layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu') (HL2)

        FLAT = layers.Flatten() (HL3)

        FC1 = layers.Dense(512, activation='relu') (FLAT)

        outputs = layers.Dense(self.nActions) (FC1)
        
        model = keras.models.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mse')

        self.q_eval = model


    def remember(self, state, action, reward, done):
        self.memory.store_transition(state, action, reward, done)
    
    def sampleForAction(self):
        return self.memory.sampleForAction()
    
    def newGame(self):
        self.memory.newGame()
        

    def choose_action(self, state):
        # state is only None at the start of each game
        if (state is None):
            action = np.random.choice(self.actionsSpace)
            print("None Sate")
            return action

        state = state[np.newaxis, :]
        
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.actionsSpace)
        else:
            actions = self.q_eval(state)
            action = np.argmax(actions)

        if (self.epsilon > self.epsilonMin):
            self.epsilon = self.epsilon * self.epsilonDec

        return action


    def learn(self):
        if self.memory.memPtr < self.batchSize + 4 and self.memory.rollOver == 0:
            return
        
        state, action, reward, newState, done = self.memory.sample_buffer(self.batchSize)
        
        # Normalize to floats between 0 and 1
        state = np.divide(state, 255, dtype=np.float32)
        newState = np.divide(newState, 255, dtype=np.float32)

        q_eval = self.q_eval(state)
        q_next = self.q_eval(newState)

        q_eval = np.array(q_eval)
        q_next = np.array(q_next)
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batchSize, dtype=np.int32)

        # Current reward + best possible action from next state - used for loss
        q_target[batch_index, action] = reward + self.gamma * np.max(q_next, axis=1) * done

        self.q_eval.fit(state, q_target, verbose=0)
    

    def save_model(self, modelFile, buffFile):
        self.q_eval.save(modelFile + ".keras")

        self.memory.saveGames(buffFile)

        with open(modelFile + "_meta.pkl", "wb") as file:
            pickle.dump([self.epsilon, self.epsilonDec, self.epsilonMin], file)
    

    def load_model(self, modelFile, buffFile=None):
        model = keras.models.load_model(modelFile + ".keras")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mse')
        self.q_eval = model
        
        if (buffFile is not None):
            self.memory.loadGames(buffFile)
        
            with open(modelFile + "_meta.pkl", "rb") as file:
                temp = pickle.load(file)
                
                self.epsilon = temp[0]
                self.epsilonDec = temp[1]
                self.epsilonMin = temp[2]