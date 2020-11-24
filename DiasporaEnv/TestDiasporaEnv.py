import gym
from numpy import mean,around
import matplotlib.pyplot as plt
import time

open('DiasporaGym/config/preprocess.json','r').read().replace("\n ","").replace("  "," ")
nbr_episodes=50
steps_per_episodes=200
rewards=[]
start=time.time()
env = gym.make("DiasporaGym:diaspora-v0")
for i_episode in range(nbr_episodes):
    print('###### Episode : {} ######'.format(i_episode))
    observation = env.reset()
    reward=0
    for t in range(steps_per_episodes):
        env.render()
        action = env.action_space.sample()
        observation = env.step(action)
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


'''
def plot():
    final_means=[]
    final_totals=[]
    for nbr_episodes in [10,20,30,40,50]:
        temp_means=[]
        temp_totals=[]
        for steps_per_episodes in [100,200,300,400,500]:
            means=[]
            totals=[]
            for i in range(10):
                mean_i,total_i=run(nbr_episodes,steps_per_episodes)
                means.append(mean_i)
                totals.append(total_i)
            print((nbr_episodes,steps_per_episodes))
            temp_means.append(mean(means))
            temp_totals.append(mean(totals))
        final_means.append(temp_means)
        final_totals.append(temp_totals)
    return final_means, final_totals

#means,totals=plot()
means=[[1.652, 3.106, 1.948, 4.452, 4.384], [5.454, 8.644, 15.9, 18.198, 20.542], [3.746, 8.298, 13.152, 16.072, 18.71], [5.11, 7.608, 11.95, 13.862, 20.73], [3.40, 7.79, 11.774, 16.128, 21.05]]
totals=[[16.514, 31.076, 19.476, 44.518, 43.818], [109.098, 172.844, 317.96, 364.0, 410.822], [112.452, 248.892, 394.606, 482.122, 561.26], [204.452, 304.36, 477.908, 554.50, 829.08], [169.97, 389.53, 588.72, 806.326, 1052.544]]
x=[100,200,300,400,500]
styles=['r--','b^-','gx-','m-','y:']

plt.plot(x,totals[0],styles[0],label='10 ep.')
plt.plot(x,totals[1],styles[1],label='20 ep.')
plt.plot(x,totals[2],styles[2],label='30 ep.')
plt.plot(x,totals[3],styles[3],label='40 ep.')
plt.plot(x,totals[4],styles[4],label='50 ep.')
plt.title('Total Reward Per Episode\n In a range of 100-500 iter per episode, runned 10 times per episode')
plt.xlabel('Steps per Episode')
plt.ylabel('Total Reward')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('total_reward_per_episode.png',transparent=False)'''
