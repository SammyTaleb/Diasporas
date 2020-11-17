import gym
import DiasporaGym
from keras_preprocessing.sequence import pad_sequences

open('DiasporaGym/config/preprocess.json','r').read().replace("\n ","").replace("  "," ")
env = gym.make("DiasporaGym:diaspora-v0")
print(env.observation_space)
print(env.action_space)
print(env.reward_range)
print(env.metadata)
env.reset()