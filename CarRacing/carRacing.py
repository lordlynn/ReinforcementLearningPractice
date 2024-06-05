import keras
import keras.backend
import numpy as np
import gymnasium as gym
import Agent


loadModelFile = "DQN_MC.keras"
loadBuffFile = "RB"
modelSaveFile = "DQN_MC.keras"
buffFile = "RB"
env = gym.make("MountainCar-v0", max_episode_steps=100)

# Potentialy make the memSize 100k instead of 10k. See how this goes first
agent = Agent.Agent(gamma=0.99, epsilon=1.0, learningRate=0.001, inputDims=2, nActions=3, memSize=100000, batchSize=64, epsilonEnd=0.010)
# agent.build_network()

agent.load_model(loadModelFile, loadBuffFile)

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

    if (i % 1 == 0 and i > 0):
        agent.save_model(modelSaveFile, buffFile)
    
    



