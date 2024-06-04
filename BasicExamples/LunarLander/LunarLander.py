import tensorflow as tf
import keras
from keras import layers

import keras.backend
import keras.optimizers
import numpy as np


import gymnasium as gym
import time


class replayBuffer(object):
    def __init__(self, maxSize, inputShape, nActions, discrete=False):
        self.memSize = maxSize
        self.discrete = discrete

        # What is the use of this??
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

class Agent(object):
    def __init__(self, alpha, gamma, nActions, epsilon, batchSize, inputDims, epsilonDec=0.996, epsilonEnd=0.01, memSize=1000000, fname='dqn_model.h5'):
        self.actionsSpace = [i for i in range(nActions)]

        self.alpha = alpha              # Learning Rate
        self.gamma = gamma              # Discount Factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilonDec = epsilonDec    # Decay rate of epsilon
        self.epsilonMin = epsilonEnd    # Minimum value of epsilon
        self.batchSize = batchSize      # Training batch size
        self.modelFile = fname          # File name to store/load model

        self.memory = replayBuffer(memSize, inputDims, nActions, discrete=True)

        self.q_eval = self.build_network(alpha, nActions, inputDims)

    def build_network(self, learningRate, nActions, input_dims):
        # inputs = layers.Input(shape=(input_dims,))

        # HL1 = layers.Dense(64, activation='relu') (inputs)
        # HL2 = layers.Dense(256, activation='relu') (HL1)
        # HL3 = layers.Dense(128, activation='relu') (HL2)
        
        # outputs = layers.Dense(nActions, activation='softmax') (HL3)
        
        # model = keras.Model(inputs=inputs, outputs=outputs)

        model = keras.Sequential([layers.Dense(256, input_shape=(input_dims,)),
                                layers.Activation('relu'),
                                layers.Dense(256),
                                layers.Activation('relu'),
                                layers.Dense(nActions)])
        
       

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss='mse')

        return model

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

        return action


    def learn(self):
        if self.memory.memPtr < self.batchSize:
            return
        
        state, action, reward, newState, done = self.memory.sample_buffer(self.batchSize)

        # Since this is discrete go back ferom one hot encoding
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
        

        if (self.epsilon > self.epsilonMin):
            self.epsilon = self.epsilon * self.epsilonDec

        

    def save_model(self):
        self.q_eval.save(self.modelFile)
        pass

    def load_model(self, modelFile):
        model = keras.models.load_model(modelFile)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        self.q_eval = model
        
        




env = gym.make("LunarLander-v2")


agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, inputDims=8, nActions=4, memSize=100000, batchSize=64, epsilonEnd=0.010)

agent.load_model("./model/LunarLander_420Epochs.h5")

scores = []
eps_history = []
n_games = 500
for i in range(n_games):
    done = False
    score = 0
    observation, info = env.reset()

    while not done:

        action = agent.choose_action(observation)
        newObservation, reward, done, truncated, info = env.step(action)

        if (truncated):
            done = True

    
        score += reward
        agent.remember(observation, action, reward, newObservation, done)

        observation = newObservation

        agent.learn()
        keras.backend.clear_session()
    

    scores.append(score)

    avg_scores = np.mean(scores[max(0, i-20):i+1])

    print(f"Episode {i}\tScore {score:.2f}\tAverage Score {avg_scores:.2f}")

    if (i % 10 == 0 and i > 0):
        agent.save_model()



