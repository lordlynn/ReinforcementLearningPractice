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
num_inputs = 2                                                      # [position, velocity]
num_actions = 3                                                     # [0 - accelerate left, 1 - don't accelerate, 2 - accelerate right]
num_hidden = 128                                                    # Number of nodes in hidden layer

# -------------------- Variables for training --------------------
optimizer = None
lossFunction = None
action_history = []
critic_history = []
rewards_history = []
running_reward = 0
episode_count = 0
eps = np.finfo(np.float32).eps.item()  
gamma = 0.99

# --------------- Gymnaisum environment variables ----------------
max_steps_per_episode = 4000
env = gym.make("MountainCar-v0", max_episode_steps=max_steps_per_episode)
observation, info = env.reset()
action_space = [0, 1, 2]


def setupNetwork():
    global optimizer, lossFunction
    global model

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    lossFunction = keras.losses.Huber()

    # Input layer for CartPole observations 
    inputs = layers.Input(shape=(num_inputs,))

    # Hidden layer
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    
    # Actor and critic outputs
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    # Overall model
    model = keras.Model(inputs=inputs, outputs=[action, critic])
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

        # Use gradient tape to perform training
        with tf.GradientTape() as tape:
            
            for stp in range(max_steps_per_episode):
                # 1.) Pass state to the model and get an action and critic 
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, axis=0)
                actions, critic = model(state)

                # 2.) Select action
                action = np.random.choice(action_space, p=np.array(actions[0]))

                # 3.) Move to next step using action
                state, reward, terminated, truncated, info = env.step(action)
                
            
                # 4.) Record results in episode lists
                action_history.append(tf.convert_to_tensor(tf.math.log(actions[0][action]), dtype=np.float32))
                critic_history.append(tf.convert_to_tensor(critic[0][0], dtype=np.float32))
                
                
                # Add speed modifier to reward   
                reward += (abs(state[1]) / 0.07)
            
                rewards_history.append(reward)
                
               
                # 5.) Break out of episode if end is reached
                if (truncated):
                    break
                elif (terminated):
                    rewards_history[-1] += 1000
                    break
            
            

            ############# Train after episode #############
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)


            # Normalize - standard normal
            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_history, critic_history, returns)
            actor_losses = []
            critic_losses = []

            # log_prob = action_history, value = critic_history, ret = returns
            for log_prob, value, ret in history:
                # the critic estimated that we would get a reward (value), but we got a actual
                # reward of (ret) for taking action with (log_probability).
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.

                # actor loss = action_prob * (return - critic)
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
                

                # Use huber loss for critic. Use critic history and rewards earned
                critic_losses.append(
                    lossFunction(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            # Combine both actor and critic losses
            loss_value = sum(actor_losses) + sum(critic_losses)
            
            # Calculate the gradient
            grads = tape.gradient(loss_value, model.trainable_variables)
            
            # update model
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


        # Log details
        episode_count += 1
        reward_progress.append(sum(rewards_history))
        avgReward = np.mean(reward_progress[max(0, episode_count-20):episode_count])
        
        
        print(f"Episode {episode_count}\tReward: {sum(rewards_history):.2f}\tAVG Reward: {avgReward:.2f}\tLoss: {loss_value:.2f}")

        if (episode_count % 10 == 0):
            keras.models.save_model(model, "MountainCar.keras")

        if (episode_count > 100):
            break
        





setupNetwork()

model = keras.models.load_model("MountainCar.keras")
model.compile(optimizer=optimizer, loss=lossFunction)

trainNetwork()

plt.figure(1)
plt.plot(reward_progress)
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.show()

