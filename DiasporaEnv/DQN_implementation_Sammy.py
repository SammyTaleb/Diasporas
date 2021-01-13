import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Dropout
from tensorflow.keras.models import Model, Sequential
import time



Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    '''
    Experience Replay Class
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, replay_memory_size, min_replay_memory_size, gamma, update_target):
        # Main model
        
        self.gamma=gamma
        self.update_target=update_target
        
        self.min_replay_memory_size=min_replay_memory_size
        self.model = self.create_model((1,160,))

        # Target network
        self.target_model = self.create_model((1,160,))
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = ReplayMemory(replay_memory_size)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    @staticmethod
    def create_model(tensor_shape):
        model = Sequential()
        
        model.add(Input(shape=tensor_shape))
        model.add(Dense(units=30, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=20, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=8, activation='relu'))
        
        model.compile(loss=tf.keras.losses.mean_squared_error,
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (state, action, reward, next state)
    def update_replay_memory(self, transition):
        self.replay_memory.push(*transition)

    # Trains main network every step during episode
    def train(self, terminal_state,batch_size):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        
        
        # Get a minibatch of random samples from memory replay table
        minibatch = self.replay_memory.sample(batch_size=batch_size)
        
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = [np.array([np.array(transition[0]).reshape((1,160))]) for transition in minibatch]
        current_qs_list=[]
        for cur_state in current_states:
            current_qs_list.append(self.model.predict(cur_state,use_multiprocessing=True))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = [np.array([np.array(transition[3]).reshape((1,160))]) for transition in minibatch]
        future_qs_list=[]
        for new_cur_state in new_current_states:
            future_qs_list.append(self.target_model.predict(new_current_states,use_multiprocessing=True))
       
        X = []
        y = []
        
        
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):

            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.gamma * max_future_q
           

            # Update Q value for given state
            current_qs = current_qs_list[index][0][0]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append([current_qs])
        
        
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=1, shuffle=False,use_multiprocessing=True)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.update_target:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array([np.array(state).reshape((1,160))]),use_multiprocessing=True)


def processState(env,state,action=None):
    final_state=[]
    for ele in state:
        try:
            a=ele.tolist()[0]
            final_state=final_state+a
        except:
            final_state=final_state+[ele]
        
    final_state=np.array([final_state])
    actions=[0]*len(env.actionHash)
    if action:
        actions[action]=1
    actions=np.array([actions])
    f_state=np.concatenate([np.array(final_state),np.array(actions)],axis=1)
    return f_state
    
    

############  PARAMETERS ##########
EPISODES=10
STEPS_PER_EPISODE=100
EPSILON=0.9
EPSILON_DECAY=0.9
MIN_EPSILON=0.2

########### DDQN HYPERPARAMETERS #############
REPLAY_MEMORY_SIZE=1000
MIN_REPLAY_MEMORY_SIZE_FOR_TRAINING=40
DISCOUNT_FACTOR=0.9
UPDATE_TARGET_EVERY=40
BATCH_SIZE=40


env = gym.make("DiasporaGym:diaspora-v0")
agent=DQNAgent(REPLAY_MEMORY_SIZE,MIN_REPLAY_MEMORY_SIZE_FOR_TRAINING,DISCOUNT_FACTOR,UPDATE_TARGET_EVERY)

ep_rewards=[]
for i in range(EPISODES):
    current_state = env.reset()
    current_state=processState(env,current_state)
    tot_reward=0
    done = False
    for i in range(STEPS_PER_EPISODE):

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > EPSILON:
            # Get action from Q table
            qs=agent.get_qs(current_state)
            action = np.argmax(qs)
        else:
            # Get random action
            action = env.action_space.sample()

        new_state, reward, done = env.step(action)
        new_state=processState(env,new_state)

        # Transform new continous state to new discrete state and count reward
        tot_reward += reward


        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action,reward,new_state))
        agent.train(done,BATCH_SIZE)

        current_state = new_state

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(tot_reward)
    print('Total Reward of the Episode',tot_reward)
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON, EPSILON)

