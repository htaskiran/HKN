#!/usr/bin/env python
# coding: utf-8

# In[18]:


import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import random
from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np
from sklearn.preprocessing import Normalizer
#Import libraries



#Data visualization

import matplotlib.pyplot as plt

#Data Manipulation
import pandas as pd
import numpy as np

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from tensorflow.keras.models import load_model
# load model
model = load_model('model.h5')
# summarize model.
model.summary() 
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import random
from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np
from sklearn.preprocessing import Normalizer
#Import libraries

# load model
model = load_model('model.h5')
# summarize model.
model.summary() 

#ENVIRONMET
#################################################################################################################################
class CiruitTrain(Env):
    def __init__(self):
        
        self.state_space = Box(low=-1, high=1, shape=(8,))
        
         
        self.action_space = Box(low=-1, high=1, shape=(8,))
        
        # current state 
        self.state =np.array([[0.202547 ,  1.035784  ,  0.69495 ,  0.041419 ,  0.150493  , -0.39333   , -0.426001,  0.708016]])  
  
        self.fom=[]
        #self.reward=0
        self.gain=[]
        self.bw=[]
        self.action=[]
        self.param=[]
        self.rr= []
        
    def step(self, action):
        done = False
        info={}
        obs = self.state + action
        
        gain,bw = model.predict(obs)
        
        performance= gain*bw
        
        self.gain.append(gain)
        self.bw.append(bw)
        self.fom.append(performance)
        self.action.append(action)
        self.param.append(obs)
        #self.gain[-1] < 0.45 and self.gain[-1] > 0.035:
        reward= performance-self.fom[-2]
        if  self.bw[-1] > 0.1:
            reward= reward+100
        else :
            reward=reward-25
        self.rr.append(reward)
        return obs, reward ,done, info
    
    def reset(self):
        self.state =np.array([[0.202547 ,  1.035784  ,  0.69495 ,  0.041419 ,  0.150493  , -0.39333   , -0.426001,  0.708016]])     
        return self.state  
    
def maximum(a, b):
     
    if a >= b:
        return a
    else:
        return b



env =  env=CiruitTrain() 

num_states = env.state_space.shape[0]
#print("Size of State Space ->  {}".format(num_states))
num_states = env.state_space.shape[0]
#print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
#print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

#print("Max Value of Action ->  {}".format(upper_bound))
#print("Min Value of Action ->  {}".format(lower_bound))

#ENVIRONMET BİTİŞİ
###################################################################################################################################

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.4, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

            

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state  = np.zeros((self.buffer_capacity, num_states))
        self.action  = np.zeros((self.buffer_capacity, num_actions))
        self.reward  = np.zeros((self.buffer_capacity, 1))
        self.next_state  = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, state, action, reward, next_state):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state [index] =  state
        self.action [index] =  action
        self.reward [index] =  reward
        self.next [index] = next_state
        

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

        

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
     
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

     
    #outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)
    state_out = layers.Dense(128, activation="relu")(state_out)
    

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(128, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model



def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


std_dev = 0.5
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.0001
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

 
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.01

buffer = Buffer(100000, 64)


# In[19]:


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
env.fom.append(0)
# Takes about 4 min to train
i=0
total_episodes=20


 
episodic_reward = 0
prev_state = env.state
episodic_reward = 0
env.bw.append(0)
env.gain.append(0)
     
while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        #tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)  
        action = policy(prev_state, ou_noise)
        # Recieve state and reward from environment.
         
        prev_state, reward, done, info = env.step(action)
        #buffer.record((  action, reward, prev_state    ))
        
        
        #print(state)
        episodic_reward += reward
        ep_reward_list.append(reward)
        print('Episode * ', i ,'* Avg Reward is ==> ',episodic_reward)
        #buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        
        
        print('BW',env.bw[i])
        print('gain',env.gain[i])
        #print(action)
        i=i+1
        # End this episode when `done` is True
        if i==30000:
            break
            


    

#Mean of last 40 episodes
#avg_reward = np.mean(ep_reward_list[-40:])
#print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
   #avg_reward_list.append(avg_reward)
    
# Plotting graph
# Episodes versus Avg. Rewards
 
 


# In[23]:


x = np.arange(0, len(env.bw[0:3000] ))
y=env.bw[0:3000] 
 
# plotting
plt.title("BW graph")
plt.xlabel("Number of Simulations")
plt.ylabel("Bandwidth")
plt.plot(x, y, color ="red")
plt.show()
x = np.arange(0, len(env.bw ))
y=env.bw 
 
# plotting
x = np.arange(0, len(env.bw ))
y=env.gain
 
# plotting
plt.title("Gain graph")
plt.xlabel("Number of Simulations")
plt.ylabel("Gain")
plt.plot(x, y, color ="b")


# In[ ]:




