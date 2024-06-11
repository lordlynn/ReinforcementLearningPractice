import numpy as np
import gymnasium as gym
import Agent
import ReplayBuffer

def rgb2gray(rgb):
    return np.array(np.dot(rgb[...,:3], [0.3333, 0.3333, 0.3333]), dtype=np.uint8)


def main(n_epochs):
    # Run training loop
    scores = []
    n_games = n_epochs

    for i in range(n_games):
        done = False
        score = 0
        observation, info = env.reset()
        step = 1

        while not done:
            state = agent.memory.lastFrames()

            action = agent.choose_action(state)

            newObservation, reward, done, truncated, info = env.step(action)
            
            if (truncated):
                done = True

            agent.remember(rgb2gray(observation), action, reward, rgb2gray(newObservation), done)
            
            score += reward
            observation = newObservation
            

            step += 1

        
        scores.append(score)

        avg_scores = np.mean(scores[max(0, i-5):i+1])

        print(f"Episode {i}\tScore {score:.2f}\tAverage Score {avg_scores:.2f}")
    
    



if __name__ == "__main__":
    loadModelFile = "./checkpoints/DQN_NEW_75.keras"

    env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
    agent = Agent.Agent(gamma=0.99, epsilon=0.000, learningRate=0.001, inputDims=(96,96,4), nActions=5, memSize=20000, batchSize=64, epsilonEnd=0.000)
    


    agent.load_model(loadModelFile)

    main(15)
