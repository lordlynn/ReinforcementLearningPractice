# https://keras.io/examples/rl/actor_critic_cartpole/
import os
os.environ["KERAS_BACKEND"] = "tensorflow"      # Set keras bacedn to ensure tesnorflow is used for computations

import numpy as np
import gymnasium as gym

import keras
from keras import ops
from keras import layers
import tensorflow as tf



# ----------------- Network Architecure variables ----------------
action = None
critic = None
model = None
num_inputs = 4                                                      # [position, velocity, pole angle, pole angular velocity]
num_actions = 2                                                     # [0 - left, 1 - right]
num_hidden = 128                                                    # Number of nodes in hidden layer


# --------------- Gymnaisum environment variables ----------------
seed = 42
gamma = 0.99                                                        # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=seed)
eps = np.finfo(np.float32).eps.item()                               # Smallest number such that 1.0 + eps != 1.0


# -------------------- Variables for training --------------------
optimizer = None
huber_loss = None
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0


def setupNetwork():
    global optimizer, huber_loss
    global action, critic, model

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()

    # Input layer for CartPole observations 
    inputs = layers.Input(shape=(num_inputs,))

    # Hidden layer
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    
    # Actor and critic outputs
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    # Overall model
    model = keras.Model(inputs=inputs, outputs=[action, critic])


def trainNetwork():
    global episode_count, running_reward
    global action_probs_history, critic_value_history, rewards_history
    
    # Main loop for training (Each iteration 1 Episode)
    while True:  
        state, info = env.reset()
        episode_reward = 0
        
        
        with tf.GradientTape() as tape:

            # Iterate through an episode
            # Basically use the current model and record:
            #       - Actions taken
            #       - Rewards earned
            #       - Critic values
            #       - Running Reward (scalar)
            for timestep in range(1, max_steps_per_episode):
        
                # Convert state to a tesnor so it can be used by the model
                state = ops.convert_to_tensor(state)
                state = ops.expand_dims(state, axis=0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(ops.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                state, reward, terminated, truncated, _ = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward

                if terminated:
                    # print(f'Terminated at step {timestep}')
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up receiving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        print(f"running reward: {running_reward:.2f} at episode {episode_count}")

        if running_reward > 195:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break


setupNetwork()
trainNetwork()