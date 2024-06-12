import keras
import keras.backend
import numpy as np
import gymnasium as gym
import Agent



def rgb2gray(rgb):
    return np.array(np.dot(rgb[...,:3], [0.3333, 0.3333, 0.3333]), dtype=np.uint8)

def main(n_epochs):
    # Run training loop
    scores = []
    n_games = n_epochs

    for i in range(n_games):
        done = False
        score = 0
        step = 1

        # Create a new gameFrames object and setup environment
        agent.newGame()
        observation, info = env.reset()

        while not done:
            # Get the most recent sample of frames from the buffer and normalize
            state = agent.sampleForAction()
            if (state is not None):
                state = np.divide(state, 255, dtype=np.float32)

            # Use the most recent frames to predict next action
            action = agent.choose_action(state)

            # Step environment
            observation, reward, done, truncated, info = env.step(action)
            
            if (truncated):
                done = True

            # Add data to buffer
            agent.remember(rgb2gray(observation), action, reward, done)
            
            score += reward
            
            # Train every 2 steps through environment
            if (step % 2 == 0):

                agent.learn()
                keras.backend.clear_session()

            step += 1
        
        with open("scores.txt", "a") as fp:
            fp.write(str(score)+"\n")

        scores.append(score)

        avg_scores = np.mean(scores[max(0, i-20):i+1])

        print(f"Episode {i}\tScore {score:.2f}\tAverage Score {avg_scores:.2f}")

    agent.save_model(modelSaveFile, buffFile)
    
    



if __name__ == "__main__":
    loadModelFile = "DQN_NEW"
    loadBuffFile = "RB"
    modelSaveFile = "DQN_NEW"
    buffFile = "RB"

    env = gym.make("CarRacing-v2", continuous=False)

    agent = Agent.Agent(gamma=0.99, epsilon=1.00, learningRate=0.0001, inputDims=(96,96,4), nActions=5, memSize=100000, batchSize=32, epsilonEnd=0.1, epsilonDec=0.9999)

    # agent.build_network()

    agent.load_model(loadModelFile, loadBuffFile)

    main(1)
