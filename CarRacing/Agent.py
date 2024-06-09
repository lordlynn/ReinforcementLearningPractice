import keras
from keras import layers
import keras.optimizers
import numpy as np

import ReplayBuffer
import pickle


class Agent(object):
    def __init__(self, learningRate, gamma, nActions, epsilon, batchSize, inputDims, epsilonDec=0.996, epsilonEnd=0.01, memSize=1000000):
        self.actionsSpace = [i for i in range(nActions)]
        
        self.nActions = nActions                            # Number of actions
        self.inputDims = inputDims                          # Number of inputs (state)
        self.learningRate = learningRate                    # Learning Rate
        self.gamma = gamma                                  # Discount Factor
        self.epsilon = epsilon                              # Exploration rate
        self.epsilonDec = epsilonDec                        # Decay rate of epsilon
        self.epsilonMin = epsilonEnd                        # Minimum value of epsilon
        self.batchSize = batchSize                          # Training batch size

        self.memory = ReplayBuffer.ReplayBuffer(memSize, self.inputDims, nActions, discrete=True, stateType=np.uint8)
        

    def build_network(self):
        if (isinstance(self.inputDims, int)):
            inputs = layers.Input(shape=(self.inputDims,))
        else:
            inputs = layers.Input(shape=self.inputDims)


        HL1 = layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu') (inputs)
        MP1 = layers.MaxPooling2D((2, 2)) (HL1)
        HL2 = layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu') (MP1)
        MP2 = layers.MaxPooling2D((2, 2)) (HL2)
        HL3 = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu') (MP2)
        MP3 = layers.MaxPooling2D((2, 2)) (HL3)
        
        GP = layers.GlobalAveragePooling2D() (MP3)

        outputs = layers.Dense(self.nActions) (GP)
        
        model = keras.models.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mse')

        self.q_eval = model


    def remember(self, state, action, reward, newState, done):
        self.memory.store_transition(state, action, reward, newState, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.actionsSpace)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        if (self.epsilon > self.epsilonMin):
            self.epsilon = self.epsilon * self.epsilonDec

        return action


    def learn(self):
        if self.memory.memPtr < self.batchSize:
            return
        
        state, action, reward, newState, done = self.memory.sample_buffer(self.batchSize)

        # Since this is discrete go back from one hot encoding
        action_values = np.array(self.actionsSpace, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        action_indices = np.array(action_indices, dtype=np.int32)

        q_eval = self.q_eval(state)
        q_next = self.q_eval(newState)

        q_eval = np.array(q_eval)
        q_next = np.array(q_next)
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batchSize, dtype=np.int32)

        # Current reward + best possible action from next state - used for loss
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        self.q_eval.fit(state, q_target, verbose=0)
    

    def save_model(self, modelFile, buffFile):
        self.q_eval.save(modelFile)

        self.memory.save_buffer(buffFile)
    

    def load_model(self, modelFile, buffFile):
        model = keras.models.load_model(modelFile)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mse')
        self.q_eval = model
        
        self.memory.load_buffer(buffFile)