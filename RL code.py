# -*- coding: utf-8 -*-
"""

@author: KETHAN

install gym,stable_baselines3 from prompt before executing the code.
"""

import gym
import os
import math
import numpy as np
from stable_baselines3 import PPO
import random

class WirelessEnv(gym.Env):
    def __init__(self, gamma, rho, doppler_frequency, num_users, interference_factor, max_energy):
        super(WirelessEnv, self).__init__()
        

        # Define the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # Define the action space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Define other parameters for the environment
        self.distance = 1000  # Distance between transmitter and receiver (m)
        self.frequency = 2.4e9  # Frequency of operation (Hz)
        self.bandwidth = 20  # System bandwidth (MHz)
        self.transmit_power = 0.0  # Initial transmit power
        self.shadowing_std = 4.0  # Standard deviation of shadowing (in dB)
        self.fading_std = 2.0  # Standard deviation of fading (in dB)
        self.doppler_frequency = doppler_frequency  # Doppler frequency (Hz)
        self.transmit_power = self.calculate_initial_transmit_power(doppler_frequency) 
        self.num_users = num_users  # Number of interfering users
        self.interference_factor = interference_factor  # Interference factor
        self.gt = 40
        self.gr = 40
        # Individual mobility model parameters for the main user
        self.gamma = gamma
        self.rho = np.random.uniform(0, 1)

        # Individual mobility model parameters for interfering users
        self.users_gamma = [np.random.uniform(0, 1) for _ in range(self.num_users)]
        self.users_rho = [np.random.uniform(0, 1) for _ in range(self.num_users)]

        # Time-dependent parameters
        self.time_step = random.uniform(0.5, 1.5)  # Time step duration in seconds
        self.user_speed = self.distance / self.time_step
        self.user_position = 0  # Initialize the user's position
        self.max_energy = max_energy
        self.energy_consumption = 0.0

    
    def calculate_initial_transmit_power(self, doppler_frequency):
        transmit_power = 0.0
        doppler_shift = math.sqrt(1 - (self.doppler_frequency / self.frequency) ** 2)
        if doppler_shift > 0:
            transmit_power = self.transmit_power / doppler_shift
        return transmit_power

    def step(self, action):
        # Map the continuous action to the transmit power
        transmit_power = self.map_action(action[0])

        # Limit the transmit power to a certain value (e.g., between 0 and 100)
        transmit_power = np.clip(transmit_power, 0, 2)

        # Update the system state based on the chosen action
        self.update_system_state(transmit_power)

        # Calculate the observation based on the current system state
        observation = self.calculate_observation()

        # Calculate the reward based on the data rate achieved
        reward = self.calculate_reward()

        # Set the done flag based on a termination condition (e.g., end of episode)
        done = self.calculate_data_rate(self.calculate_path_loss(self.distance, self.frequency), self.bandwidth) > 80.0
        # Calculate the energy consumption based on the transmit power and time step
        energy = self.transmit_power * self.time_step
        self.energy_consumption += energy

        # Check if the energy consumption exceeds the maximum energy limit
        if self.energy_consumption > self.max_energy:
            done = True
            reward = -1  # Penalize for exceeding the energy constraint
        else:
            done = False
            reward = self.calculate_reward()

        # Set the info dictionary (optional)
        info = {
            'transmit_power': transmit_power
        }

        return observation, reward, done, info

    def reset(self):
        # Reset the system state to the initial configuration
        self.transmit_power = 0.0
        self.energy_consumption = 0.0

        

        # Calculate the initial observation based on the current system state
        observation = self.calculate_observation()

        return observation

    def map_action(self, action):
        # Map the continuous action to the transmit power
        transmit_power = action * 100.0

        return transmit_power

    def update_system_state(self, transmit_power):
        # Update the system state based on the chosen action
        self.transmit_power = transmit_power

        # Update user mobility
        self.update_user_mobility()
        self.update_interfering_users_mobility()

    def update_user_mobility(self):
        # Update the user's position based on an individual mobility model
        # The individual mobility model can use parameters gamma and rho to estimate the probability of the next step
        # Update the distance based on the user's movement
        distance_change = self.user_speed * self.time_step
        self.distance += np.random.choice([-1, 1], p=[self.gamma, 1 - self.gamma]) * np.random.exponential(
            self.rho) * distance_change

    def update_interfering_users_mobility(self):
        # Update the interfering users' positions based on individual mobility models
        for i in range(self.num_users):
            user_speed = self.user_speed * (i + 2)  # Example: Different speeds for interfering users
            distance_change = user_speed * self.time_step
            self.distance += np.random.choice([-1, 1], p=[self.users_gamma[i], 1 - self.users_gamma[i]]) * \
                             np.random.exponential(self.users_rho[i]) * distance_change

    def calculate_observation(self):
        # Calculate the observation based on the current system state
        normalized_power = self.transmit_power #/ 100.0
        normalized_distance = self.distance #/ 10000.0
        normalized_frequency = self.frequency #/ 1e9
        normalized_interference = self.calculate_interference() #/ 100.0
        normalized_signal_power = self.calculate_signal_power() #/ 100.0
        normalized_noise_power = self.calculate_noise_power() #/ 100.0

        observation = np.array(
            [normalized_power, normalized_distance, normalized_frequency, normalized_interference,
             normalized_signal_power, normalized_noise_power])

        return observation

    def calculate_reward(self):
       # Calculate the data rate and path loss
        path_loss = self.calculate_path_loss(self.distance - self.user_position, self.frequency)
        data_rate = self.calculate_data_rate(path_loss, self.bandwidth)

        # Calculate the reward based on the square of the data rate divided by the path loss
        reward = (data_rate ** 2) / path_loss

        return reward

    def calculate_path_loss(self, distance, frequency):
        # Calculate the path loss based on a model
        c = 3e8  # Speed of light (m/s)
        lambda_ = c / frequency  # Wavelength (m)
        path_loss = (4 * np.pi * distance / lambda_) ** 2

        return path_loss

    def calculate_data_rate(self, path_loss, bandwidth):
        # Calculate the signal power and noise power
        signal_power = self.calculate_signal_power()
        noise_power = self.calculate_noise_power()
        interference_power = self.calculate_interference_power()

        # Calculate the SNR from the signal power and noise power
        snr = signal_power / noise_power
        sinr = interference_power / noise_power

        # Calculate the data rate based on the path loss, bandwidth, and SINR
        #data_rate = bandwidth * np.log2(1 + (snr/(1+sinr)))
        data_rate = bandwidth * np.log2(1.0 + (snr / (1.0 + sinr)))

        if np.isnan(data_rate) or np.isinf(data_rate) or data_rate < 0:
            data_rate = 0.0

        return data_rate
    

    def calculate_interference(self):
        # Calculate the interference based on the interfering users' positions and distances
        interference = 0.0
        for i in range(self.num_users):
            user_distance = self.distance + (i + 1) * 100  # Example: Different distances for interfering users
            user_path_loss = self.calculate_path_loss(user_distance, self.frequency)
            user_interference = self.interference_factor * (1 - user_path_loss)
            interference += user_interference

        return interference

    def calculate_interference_power(self):
        # Calculate the interference power based on the interfering users' positions and distances
        interference_power = 0.0
        for i in range(self.num_users):
            user_distance = self.distance + (i + 1) * 100  # Example: Different distances for interfering users
            user_path_loss = self.calculate_path_loss(user_distance, self.frequency)
            user_signal_power = self.calculate_signal_power()
            user_interference_power = user_signal_power * self.interference_factor
            interference_power += user_interference_power

        return interference_power
    
    def calculate_signal_power(self):
        signal_power1 = self.transmit_power* self.calculate_path_loss(self.distance, self.frequency)
        signal_power2 = signal_power1 * self.gt
        signal_power3 = signal_power2 * self.gr
        shadowing = np.random.normal(loc=0.0, scale=self.shadowing_std)
        signal_power4 = signal_power3 * shadowing
        fading = np.random.rayleigh(scale=self.fading_std)
        signal_power5 = signal_power4 * fading
        signal_power = ((signal_power5 * 2) / np.pi)
        return signal_power

    def calculate_noise_power(self):
        # Calculate the noise power based on a model
        noise_power =  (4 * self.bandwidth)/(10**21)

        return noise_power

# Create the custom environment with individual mobility model parameters
gamma = np.random.uniform(0, 1)
rho = np.random.uniform(0, 1)
doppler_frequency = 1e-6
num_users = 4
interference_factor = 0.5
max_energy = 1000  # Maximum energy limit in joules
env = WirelessEnv(gamma, rho, doppler_frequency, num_users, interference_factor, max_energy)
# Create the PPO agent
model = PPO("MlpPolicy", env)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Train the agent
model.learn(total_timesteps=100)

# Evaluate the agent
num_episodes = 100
total_data_rate = 0.0

for _ in range(num_episodes):
    obs = env.reset()
    done = False
    episode_data_rate = 0.0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_data_rate += env.calculate_data_rate(env.calculate_path_loss(env.distance, env.frequency),
                                                    env.bandwidth)
        episode_transmit_power = action[0] * 100.0  # Calculate the transmit power from the action
        episode_transmit_power = np.clip(episode_transmit_power, 0, 2)  # Limit the transmit power to 0-100
        #print("Episode Transmit Power:", episode_transmit_power)

    total_data_rate += episode_data_rate

average_data_rate = total_data_rate / num_episodes

print("Average Data Rate:", average_data_rate)
import gym
import math
import numpy as np
from stable_baselines3 import PPO
import random
import matplotlib.pyplot as plt

# Rest of your code...

import matplotlib.pyplot as plt

# ...

num_episodes = 100
total_data_rate = 0.0
rewards = []  # Store rewards for each episode

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_data_rate = 0.0
    episode_rewards = []  # Store rewards for each step in the episode

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_data_rate += env.calculate_data_rate(env.calculate_path_loss(env.distance - env.user_position,
                                                                            env.frequency), env.bandwidth)
        episode_transmit_power = action[0] * 100.0  # Calculate the transmit power from the action
        episode_transmit_power = np.clip(episode_transmit_power, 0, 2)  # Limit the transmit power to 0-100
        episode_reward = (episode_data_rate ** 2) / env.calculate_path_loss(env.distance - env.user_position,
                                                                             env.frequency)
        episode_rewards.append(episode_reward)

    total_data_rate += episode_data_rate
    rewards.append(episode_rewards)

    print(f"Episode {episode+1}: Data Rate = {episode_data_rate}, Reward = {episode_rewards[-1]}")

average_data_rate = total_data_rate / num_episodes

print("Average Data Rate:", average_data_rate)

# Plotting the variation between data rate and reward function
episode_lengths = [len(r) for r in rewards]
max_length = max(episode_lengths)

plt.figure()
for r in rewards:
    r += [r[-1]] * (max_length - len(r))  # Pad shorter episodes with the last reward
    plt.plot(r)

plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Variation between Data Rate and Reward Function')
plt.show()
