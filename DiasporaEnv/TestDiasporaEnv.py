import gym
from numpy import mean,around
import time

open('DiasporaGym/config/preprocess.json','r').read().replace("\n ","").replace("  "," ")

nbr_episodes=50
steps_per_episodes=300
rewards=[]
start=time.time()
env = gym.make("DiasporaGym:diaspora-v0")
env.reset()
for i_episode in range(nbr_episodes):
    observation = env.reset()
    reward=0
    for t in range(steps_per_episodes):
        env.render()
        action = env.action_space.sample()
        observation = env.step(action)
        #print(observation[0])
        reward+=observation[1]
        print('Person : {}, Information Found: {}, Reward : {}, Continue : {}'.format(observation[0][2],observation[0][-1],observation[1],observation[2]))
    print('Total Reward Before Reseting Env : {}'.format(around(reward,2))) 
    rewards.append(reward)
rewards=around(rewards, 2)
print()
print('############### END OF RUN ################\n')
print('Rewards for each Episode, Over {} Episodes : {}'.format(nbr_episodes,rewards))
print('Total Reward : {}'.format(around(sum(rewards),2)))
print('Max Reward : {}'.format(max(rewards)))
print('Mean Rewards : {}'.format(around(mean(rewards),2)))
print('Total Time : {} sec'.format(around(time.time()-start,2)))



