import sys

import os

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('/tree/python/unityagents')

fileDir = os.path.dirname(os.path.realpath('__python__'))

print(fileDir)

#print(fileDir)

#filename = os.path.join(fileDir,'\python')

# Directory with files for Unity

filename = fileDir.replace("\DRLND_continuous-control project-pytorch","\python")

print('file',filename)

# Add to system path files for Unity
sys.path.append(filename)

#sys.path.append(r"C:\Users\pgarcia\Documents\GitHub\Udacity-Reinforcement-Learning\deep-reinforcement-learning-master\Repo-Navigation-Project1\python")

#from unityagents import UnityEnvironment

from python.unityagents import UnityEnvironment
import numpy as np

print('1')
#env = UnityEnvironment(file_name=r'C:\Users\pgarcia\Documents\GitHub\Udacity-Reinforcement-Learning\deep-reinforcement-learning-master\DRLND_continuous-control project\Reacher_Windows_x86_64\Reacher.exe')
env = UnityEnvironment(file_name=r'C:\Users\pgarcia\Documents\GitHub\Udacity-Reinforcement-Learning\deep-reinforcement-learning-master\DRLND_continuous-control project-pytorch\R_20\Reacher_Windows_x86_64\Reacher.exe')
print('2')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#env = UnityEnvironment(file_name=r'C:\Users\pgarcia\Documents\GitHub\Udacity-Reinforcement-Learning\deep-reinforcement-learning-master\DRLND_continuous-control project\Reacher_Windows_x86_64\Reacher.exe')
#env = UnityEnvironment(file_name=r'C:\Users\pgarcia\Documents\GitHub\Udacity-Reinforcement-Learning\deep-reinforcement-learning-master\DRLND_continuous-control project\Reacher_Windows_x86_64\Reacher.exe')

# get the default brain
#brain_name = env.brain_names[0]
#brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print(state_size)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
#import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

from ddpg_agent import Agent

print('state size, action size', state_size, action_size)
agent = Agent(state_size, action_size, random_seed=5)

def ddpg(n_episodes=200, max_t=1000, print_every=1, plot_every=25):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        #print('state size', state.shape)
        agent.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name] 
            next_state = env_info.vector_observations
            #reward = env_info.rewards[0] 
            reward = np.array(env_info.rewards)
            reward = reward.transpose()
            done = np.array(env_info.local_done)
            done = done.transpose()
            #print('rewards', np.shape(reward))
            #print('next states', np.shape(next_state))
            #done = env_info.local_done[0] 
            #print('state ', state[0:2,0:2])
            #print('reward size', reward[0:2])
            #print('done size', done[0:2])
            #print('t',t)
            #next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.mean(reward)
            if done[0]:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            #print('\rEpisode {}\tEpisode Score: {:.2f}'.format(i_episode, scores[i_episode-1]))
        
        if i_episode % plot_every == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(1, len(scores)+1), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()

    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
