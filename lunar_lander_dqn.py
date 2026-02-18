import time
from collections import deque, namedtuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

import utils   # requires utils.py

# =========================
# Setup
# =========================

SEED = utils.SEED
tf.random.set_seed(SEED)
np.random.seed(SEED)

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4

num_episodes = 2000
max_num_timesteps = 1000
num_p_av = 100
epsilon = 1.0

# =========================
# Environment
# =========================

env = gym.make("LunarLander-v2")
state = env.reset()

state_size = env.observation_space.shape
num_actions = env.action_space.n

print("State Shape:", state_size)
print("Number of Actions:", num_actions)

# =========================
# Networks
# =========================

q_network = Sequential([
    Input(shape=state_size),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
])

target_q_network = Sequential([
    Input(shape=state_size),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
])

optimizer = Adam(learning_rate=ALPHA)

# Copy weights to target network
target_q_network.set_weights(q_network.get_weights())

# =========================
# Replay Memory
# =========================

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
memory_buffer = deque(maxlen=MEMORY_SIZE)

# =========================
# Loss Function
# =========================

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences

    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    q_values = q_network(states)
    q_values = tf.gather_nd(
        q_values,
        tf.stack([tf.range(q_values.shape[0]),
                  tf.cast(actions, tf.int32)], axis=1)
    )

    loss = MSE(y_targets, q_values)
    return loss

# =========================
# Learning Step
# =========================

@tf.function
def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    utils.update_target_network(q_network, target_q_network)

# =========================
# Training Loop
# =========================

start = time.time()
total_point_history = []

for i in range(num_episodes):

    state = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):

        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)

        next_state, reward, done, _ = env.step(action)

        memory_buffer.append(experience(state, action, reward, next_state, done))

        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            experiences = utils.get_experiences(memory_buffer)
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Avg(Last {num_p_av}): {av_latest_points:.2f}", end="")

    if (i + 1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Avg(Last {num_p_av}): {av_latest_points:.2f}")

    if av_latest_points >= 200.0 and i >= num_p_av:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save("lunar_lander_model.h5")
        break

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({tot_time/60:.2f} min)")

env.close()
