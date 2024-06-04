import os
os.environ["KERAS_BACKEND"] = "tensorflow"      # Set keras bacedn to ensure tesnorflow is used for computations

import numpy as np
import gymnasium as gym

import keras
from keras import layers

import tensorflow as tf

# plotting the progress
import matplotlib.pyplot as plt





# ----------------- Network Architecure variables ----------------
model = None
# num_inputs = 2                                                      # [position, velocity]
# num_actions = 3                                                     # [0 - accelerate left, 1 - don't accelerate, 2 - accelerate right]
# num_hidden = 128                                                    # Number of nodes in hidden layer

# -------------------- Variables for training --------------------
optimizer = None
lossFunction = None
rewards_history = []
episode_count = 0

# --------------- Gymnaisum environment variables ----------------
max_steps_per_episode = 4000
max_episodes = 10
env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=max_steps_per_episode)
observation, info = env.reset()
action_space = [0, 1, 2]


def setupNetwork(modelName):
    global model

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    lossFunction = keras.losses.Huber()

    model = keras.models.load_model(modelName)
    model.compile(optimizer=optimizer, loss=lossFunction)




def trainNetwork():
    global episode_count, reward_progress
    global action_history, critic_history, rewards_history

    reward_progress = []
    loss_value = None

    while True:

        ############# Run an episode #############
        action_history = []
        critic_history = []
        rewards_history = []
        
        
        state, info = env.reset()
    
            
        for stp in range(max_steps_per_episode):
            # Make Predictions
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, axis=0)
            actions, critic = model(state)

        
            action = np.random.choice(action_space, p=np.array(actions[0]))

            # Step the game forward with predicted action 
            state, reward, terminated, truncated, info = env.step(action)
            
            # Add speed modifier to reward   
            reward += (abs(state[1]) / 0.07)
        
            rewards_history.append(reward)
            
            
            
            # 5.) Break out of episode if end is reached
            if (truncated):
                break
            elif (terminated):
                rewards_history[-1] += 1000
                break
                   

        # Log details
        episode_count += 1
        reward_progress.append(sum(rewards_history))
        avgReward = np.mean(reward_progress[max(0, episode_count-20):episode_count])
        
        
        print(f"Episode {episode_count}\tReward: {sum(rewards_history):.2f}\tAVG Reward: {avgReward:.2f}")

        if (episode_count > max_episodes):
            break
        





setupNetwork("./Models/MountainCar_500Epochs.keras")

trainNetwork()

plt.figure(1)
plt.plot(reward_progress)
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.show()

