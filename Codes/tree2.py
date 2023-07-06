import os
import time

import torch
import numpy as np

from copy import deepcopy

from PPO import PPO
from dqn import DQNAgent

import pickle

import argparse

from multiprocessing import Pool,set_start_method,managers

import threading
import random
#import shared_memory
#from anytree import NodeMixin, RenderTree
lst = []

class link_list():  # Add Node feature
    def __init__(self, value, next_list):
        self.value = value
        self.next_list = next_list

def parse_args():
    parser = argparse.ArgumentParser("SF")
    parser.add_argument('--idx', metavar='N', type=int, default=1,
                    help='DRL training times/epoches')
    parser.add_argument('--step', metavar='N', type=int, default=100,
                    help='attack simulation times')
    parser.add_argument("--um", metavar='N', type=int, default=1)
    parser.add_argument("--pida", metavar='%', type=float, default=0.3)
    parser.add_argument("--peda", metavar='%', type=float, default=0.3)
    parser.add_argument("--pnca", metavar='%', type=float, default=0.3)
    parser.add_argument("--adv", metavar='fgs/mim/pgd/bim', type=str, default='fgs',
                    help='drl algorithm')
    parser.add_argument("--padv", metavar='%', type=float, default=0)
    parser.add_argument("--tl", metavar='%', type=float, default=0.3)
    parser.add_argument("--alpha", metavar='%', type=float, default=1)
    parser.add_argument("--nd", metavar='N', type=int, default=20)
    parser.add_argument("--m", metavar='N', type=int, default=2)
    parser.add_argument("--drl", metavar='dqn/ppo', type=str, default='dqn',
                    help='drl algorithm')
    return parser.parse_args()

class ENV_tree2(object):
    def __init__(self, length, nagents):
        self.length = length
        self.nagents = nagents
        self.state = np.zeros((self.nagents,10),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,10),dtype=float)
        self.index = 0
        return self.state
    
    def step(self,actions,states):
        done = False
        total_action = 3*actions[0] + actions[1]
        self.index = self.index*9 + total_action
        for i in range(self.nagents):
            #exec("temp = pickle.load(open('states/s%d%d.pkl','rb'))"%(self.steps,i),globals())
            #temp = shared_memory.ShareableList(name="s%d%d"%(self.steps,i))
            #print(temp.shm.name)
            name = 's%d%d'%(self.steps,i)
            temp = states[name]
            self.state[i,self.steps] = temp[self.index]
        
        #temp_r = self.node.children[total_action].reward
               
        if self.steps == self.length-1:
            done = True
        self.steps += 1
        if done:
            r = total_action#temp_r
        else:
            r = total_action#temp_r
        return self.state, r, done

def train_ppo(states,batch_size=0,bplus=0,random_seed=0,last_evaluation_episodes=0,adv='fgs',padv=0,load=False,save=True,um=0,root=None,sroot=None):

    env_name = "SF"
    has_continuous_action_space = False
    
    nagents = 2
    
    length = 7

    # state space dimension
    state_dim = 10

    # action space dimension
    if has_continuous_action_space:
        action_dim = 3
    else:
        action_dim = 3

    ###################### logging ######################

    max_ep_len = 1000                    # max timesteps in one episode

    print_freq = max_ep_len     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)

    action_std = None

    K_epochs = 10               #10-80# update policy for K epochs
    eps_clip = 0.5              #0.1-0.6# clip parameter for PPO
    gamma = 0.9                #0.8-0.99# discount factor

    lr_actor = 0.000085       # learning rate for actor network
    lr_critic = 0.00085       #0.02-0.2# learning rate for critic network

    random_seed = random_seed         # set random seed if required (0 = no random seed)

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder


    directory = "ppo_pretrain/p_%d_%s_%.1f/"%(arglist.um,arglist.adv,arglist.padv)
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = [directory + "p_{}_{}.pth".format(random_seed,i) for i in range(nagents)]

    if random_seed:
        #print("--------------------------------------------------------------------------------------------")
        #print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)
        
    # initialize PPO agents
    agents = [PPO(nagents, state_dim, action_dim, lr_actor, lr_critic, gamma,
        K_epochs, eps_clip, has_continuous_action_space, action_std) for i in range(nagents)]
    if load:
        for i in range(nagents):
            agents[i].load(checkpoint_path[i])
    # track total training time
    start_time = time.time()

    # printing variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    reward_list = []
    env = ENV_tree2(length, nagents)

    # training loop
    for time_step in range(20):
        current_ep_reward = 0
        metric_list = []
        
        output = []
                
        for num in range(50):
            random.seed(random.random())
            state = env.reset()
            reward_ref = []
            for t in range(1, max_ep_len+1):
                
                # select action with policy
                actions = [agents[i].select_action(env.state,i) for i in range(nagents)]
                state, total_action, done = env.step(actions,states)
                # saving reward and is_terminals
                reward_ref.append(total_action)
                for i in range(nagents):
                    #agents[i].buffer.rewards.append(reward[i])
                    agents[i].buffer.is_terminals.append(done)
        
                if done:
                    break
               
            
            output.append(reward_ref)

        rewards = [[]]

        for n in output:
            temp = root
            for i in n:
                temp = temp.next_list[i]
                rewards[-1].append(temp.value)
            temp_reward = sum(rewards[-1])
            reward_list.append(temp_reward)
            rewards.append([])
            print_running_reward += temp_reward
            print_running_episodes += 1         
        
        for i in range(nagents):
            for j in range(len(rewards)):
                agents[i].buffer.rewards = agents[i].buffer.rewards + rewards[j]
            if random.random()<padv:
                adv_attack = getattr(agents[i], adv)
                adv_attack()
            else:
                agents[i].update()
        i_episode += 1
        if print_running_episodes:

            # print average reward till last episode
            end_time = time.time()
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 4)

            #print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Time Cost : {}".format(i_episode, time_step, print_avg_reward, end_time - start_time))


            print_running_reward = 0
            print_running_episodes = 0
    if save:
        for i in range(nagents):
            agents[i].save(checkpoint_path[i])

    state = env.reset()
    reward_ref = []
    for t in range(1, max_ep_len+1):        
        # select action with policy
        actions = [agents[i].select_action2(env.state,i) for i in range(nagents)]
        state, total_action, done = env.step(actions,states)
        # saving reward and is_terminals
        reward_ref.append(total_action)

        if done:
            break
    return reward_list,reward_ref

def train_dqn(states,batch_size=0,bplus=0,random_seed=0,last_evaluation_episodes=0,adv='fgs',padv=0,load=False,save=False,um=1,root=None,sroot=None):

    env_name = "SF"
    has_continuous_action_space = False
    
    nagents = 2
    
    length = 7

    # state space dimension
    state_dim = 10

    # action space dimension
    if has_continuous_action_space:
        action_dim = 3
    else:
        action_dim = 3

    ###################### logging ######################

    max_ep_len = 1000                    # max timesteps in one episode
    max_training_timesteps = length*1000   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)

    action_std = None

    gamma = 0.9                #discount factor

    lr = 0.00005#0.00005:-385.84975748242715,-386.2998786517747       # learning rate for critic network

    random_seed = random_seed         # set random seed if required (0 = no random seed)

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    train_episode = 50
    batch_size = length*train_episode

    directory = "dqn_pretrain/p_%d_%s_%.1f/"%(arglist.um,arglist.adv,arglist.padv)
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = [directory + "p_{}_{}.pth".format(random_seed, i) for i in range(nagents)]

    if random_seed:
        #print("--------------------------------------------------------------------------------------------")
        #print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)
        
    # initialize dqn agents
    agents = [DQNAgent(state_dim, action_dim, nagents, batch_size, gamma, lr) for i in range(nagents)]
    if load:
        for i in range(nagents):
            agents[i].load(checkpoint_path[i])
    # track total training time
    start_time = time.time()

    # printing variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    reward_list = []

    # training loop
    env = ENV_tree2(length, nagents)
    
    while time_step < max_training_timesteps:
        current_ep_reward = 0   
                
        random.seed(random.random())          
        
        state = env.reset()
        temp = root 
        for t in range(1, max_ep_len+1):                           
            # select action with policy
            actions = [agents[i].select_action(state, i) for i in range(nagents)]
            next_state, total_action, done = env.step(actions,states)
            # saving reward and is_terminals
            temp = temp.next_list[total_action]
            for i in range(nagents):
                agents[i].remember(deepcopy(state).flatten(), deepcopy(actions[i]), deepcopy(temp.value), deepcopy(next_state).flatten(), done)   
            state = deepcopy(next_state)
            time_step += 1
            current_ep_reward += temp.value
            if done:
                break

        reward_list.append(current_ep_reward)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

        if print_running_episodes >= train_episode:
        #if print_running_episodes % train_episode == 0:
            for i in range(nagents):
                if random.random()<padv:
                    adv_attack = getattr(agents[i], adv)
                    adv_attack()
                else:
                    agents[i].update()

        if print_running_episodes % train_episode == 0:
            # print average reward till last episode
            end_time = time.time()
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 4)

            #print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Time Cost : {}".format(i_episode, time_step, print_avg_reward, end_time - start_time))


            print_running_reward = 0
            print_running_episodes = 0
        current_ep_reward = 0   
                
        random.seed(random.random())          
        
    if save:
        for i in range(nagents):
            agents[i].save(checkpoint_path[i])

    for i in range(nagents):
        agents[i].is_training = False
    state = env.reset()
    reward_ref = []
    for t in range(1, max_ep_len+1):                           
        # select action with policy
        actions = [agents[i].select_action(state, i) for i in range(nagents)]
        next_state, total_action, done = env.step(actions,states)
        # saving reward and is_terminals
        reward_ref.append(total_action)
        state = deepcopy(next_state)
        if done:
            break
    return reward_list,reward_ref

if __name__ == "__main__":
    set_start_method('spawn')# good solution !!!!
    arglist = parse_args()
    root = pickle.load(open("reward_trees/r_%d_%.1f_%.1f_%.1f_%.2f_%d_%.1f.pkl"%(arglist.um,arglist.pida,arglist.peda,arglist.pnca,arglist.tl,arglist.nd,arglist.alpha),"rb"))
    rewards_list = []
    action_list = []
    if arglist.drl == 'ppo':
        train = train_ppo
    else:
        train = train_dqn

    states = {}
    for i in range(7):
        for j in range(2):
            name = 's%d%d'%(i,j)
            states[name] = pickle.load(open('states/s_%d_%.1f_%.1f_%.1f_%.2f_%d_%.1f/s%d%d.pkl'%(arglist.um,arglist.pida,arglist.peda,arglist.pnca,arglist.tl,arglist.nd,arglist.alpha,i,j),'rb'))

    for i in range(arglist.idx*arglist.step,(arglist.idx+1)*arglist.step):
        reward_list,actions = train(states,random_seed=i,adv=arglist.adv,padv=arglist.padv,save = True,um=arglist.um,root=root)
        rewards_list.append(reward_list)
        action_list.append(actions)
    directory = "rewards/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = "actions/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(rewards_list,open("rewards/r_%s_%d_%s_%.1f.pkl"%(arglist.drl,arglist.um,arglist.adv,arglist.padv),"wb"))
    pickle.dump(action_list,open("actions/a_%s_%d_%s_%.1f.pkl"%(arglist.drl,arglist.um,arglist.adv,arglist.padv),"wb"))
    #os.system('say "you program is finishied"')