# https://keras.io/examples/rl/actor_critic_cartpole/
import os
os.environ["KERAS_BACKEND"] = "tensorflow"      # Set keras bacedn to ensure tesnorflow is used for computations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import gymnasium as gym

import keras
from keras import layers

from keras import ops
import tensorflow as tf

# plotting the progress
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------- Network Architecure variables ----------------
model = None
num_inputs = 4                                                      # [position, velocity, pole angle, pole angular velocity]
num_actions = 2                                                     # [0 - left, 1 - right]
num_hidden = 128                                                    # Number of nodes in hidden layer


# --------------- Gymnaisum environment variables ----------------
seed = 42
max_steps_per_episode = 1500
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=max_steps_per_episode)
observation, info = env.reset(seed=seed)

eps = np.finfo(np.float32).eps.item()  
gamma = 0.99


# -------------------- Variables for training --------------------
optimizer = None
lossFunction = None
action_history = []
critic_history = []
rewards_history = []
running_reward = 0
episode_count = 0


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
    # model.compile(optimizer=optimizer, loss=lossFunction)


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
                # action = env.action_space.sample()

                state = ops.convert_to_tensor(state)
                state = ops.expand_dims(state, axis=0)
                actions, critic = model(state)

                # 2.) Select action
                action = np.random.choice([0, 1], p=np.array(actions[0]))

                # 3.) Move to next step using action
                state, reward, terminated, truncated, info = env.step(action)
                
            
                # 4.) Record results in episode lists
                action_history.append(ops.convert_to_tensor(ops.log(actions[0, action]), dtype=np.float32))
                critic_history.append(ops.convert_to_tensor(critic[0][0], dtype=np.float32))
                rewards_history.append(reward)
                
                # 5.) Break out of episode if end is reached
                if (terminated or truncated):
                    # print("Terminated")
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
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

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

                # actor loss = -action_prob * (return - critic)
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # Use huber loss for critic. Use critic history and rewards earned
                critic_losses.append(
                    lossFunction(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
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
        print(f"Episode {episode_count}\tReward: {sum(rewards_history)}\tLoss: {loss_value:.2f}")

        if sum(rewards_history) > 1250:                  # Condition to consider the task solved
            print(f"Solved at episode {episode_count}")
            break
        
        reward_progress.append(sum(rewards_history))





setupNetwork()
trainNetwork()

plt.figure(1)
plt.plot(reward_progress)
plt.xlabel("Episode")
plt.ylabel("Steps survived")

plt.show()