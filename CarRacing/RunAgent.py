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

            observation = newObservation


            step += 1
        
        scores.append(score)

        avg_scores = np.mean(scores[max(0, i-5):i+1])

        print(f"Episode {i}\tScore {score:.2f}\tAverage Score {avg_scores:.2f}")
    
    



if __name__ == "__main__":
    loadModelFile = "./checkpoints/DQN_CR_LargeBuff_300.keras"

    env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
    agent = Agent.Agent(nActions=5, running=True)

    agent.load_model(loadModelFile)

    main(15)
