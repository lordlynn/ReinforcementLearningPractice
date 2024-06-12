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
        observation, info = env.reset()
        temp = rgb2gray(observation)

        temp = np.expand_dims(temp, axis=2)
        state = temp
        state = np.append(state, temp, axis=2)
        state = np.append(state, temp, axis=2)
        state = np.append(state, temp, axis=2)

        while not done:
            

            action = agent.choose_action(state) 

            observation, reward, done, truncated, info = env.step(action)
            
            if (truncated):
                done = True

            state = state[:,:,1:]
            state = np.append(state, np.expand_dims(rgb2gray(observation), axis=2), axis=2)
            
            score += reward


        
        scores.append(score)

        avg_scores = np.mean(scores[max(0, i-5):i+1])

        print(f"Episode {i}\tScore {score:.2f}\tAverage Score {avg_scores:.2f}")
    
    



if __name__ == "__main__":
    loadModelFile = "./DQN_NEW.keras"

    env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
    agent = Agent.Agent(nActions=5, running=True)
    agent.load_model(loadModelFile)

    main(15)
