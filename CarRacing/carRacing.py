import keras
import keras.backend
import numpy as np
import gymnasium as gym
import Agent


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
            action = agent.choose_action(observation)

            newObservation, reward, done, truncated, info = env.step(action)

            if (truncated):
                done = True

            score += reward
            agent.remember(observation, action, reward, newObservation, done)

            observation = newObservation
            
            # Train every 4 steps through environment
            if (step % 4 == 0):
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
    loadModelFile = "DQN_MC.keras"
    loadBuffFile = "RB"
    modelSaveFile = "DQN_MC.keras"
    buffFile = "RB"

    env = gym.make("CarRacing-v2", max_episode_steps=200, continuous=False)

    agent = Agent.Agent(gamma=0.99, epsilon=0.010, learningRate=0.001, inputDims=(96,96,3), nActions=5, memSize=20000, batchSize=64, epsilonEnd=0.010)

    # agent.build_network()

    agent.load_model(loadModelFile, loadBuffFile)

    main(5)