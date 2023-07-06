import os
import math
import numpy as np

import networkx as nx
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from copy import deepcopy

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
    parser.add_argument("--um", metavar='N', type=int, default=0)
    parser.add_argument("--pida", metavar='%', type=float, default=0.3)
    parser.add_argument("--peda", metavar='%', type=float, default=0.3)
    parser.add_argument("--pnca", metavar='%', type=float, default=0.3)
    parser.add_argument("--tl", metavar='%', type=float, default=0.3)
    parser.add_argument("--alpha", metavar='%', type=float, default=1)
    parser.add_argument("--nd", metavar='N', type=int, default=20)
    parser.add_argument("--m", metavar='N', type=int, default=2)
    parser.add_argument("--drl", metavar='dqn/ppo', type=str, default='dqn',
                    help='drl algorithm')
    return parser.parse_args()

class Landmark():
    def __init__(self,ID,init_bl):
        super(Landmark, self).__init__()
        self.ID = ID #animal ID
        self.temp_0 = np.random.normal(38,1,1)[0] #temperature
        self.temp = deepcopy(self.temp_0)
        self.hb = hbs[0,ID] #heart beat
        self.vel = vels[0,ID]
        self.tl = init_bl
        if self.ID<5:#100:
            self.bl = 1#lower the battery level
        else:
            self.bl = self.tl+self.ID*170/5000000#self.ID*
        self.t = 0 #time stamp with sec. as unit
        self.data = [self.ID, self.temp, self.hb, self.vel, self.bl,self.t]
        
    def reset(self):
        self.temp = deepcopy(self.temp_0)
        self.hb = hbs[0,self.ID] #heart beat
        self.vel = vels[0,self.ID]
        if self.ID<5:#100:
            self.bl = 1#lower the battery level
        else:
            self.bl = self.tl+self.ID*170/5000000#self.ID*
        self.t = 0 #time stamp with sec. as unit
        self.data = [self.ID, self.temp, self.hb, self.vel, self.bl, self.t]

# properties of agent entities
class Agent():
    def __init__(self,ID,sensor_number):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # action
        self.action = 0
        # pos
        self.pos = 0
        # script behavior to execute
        self.action_callback = None

        self.ID = ID #gateway ID
        self.sensor_number = sensor_number
        self.database = np.full((self.sensor_number,6),-1, dtype=float)
        #animal ID,[sender ID,HR,AT,M,BL,T,D]
        self.total_database = np.zeros((self.sensor_number,4),dtype=float)#animal ID,[step,temperature,moving activity,heart rate]
        #self.total_database[:,3]=np.inf
        self.new_database = np.full((self.sensor_number,5),-1,dtype=float) #temp, hb, mv, bl, t

        # 1 - below normal, 2 - normal, 3 - above normal
        s = (self.sensor_number,3)
        self.evidence = np.ones(s)
        self.belief = np.zeros(s)
        self.uncertainty = np.zeros(self.sensor_number)
        self.vacuity0 = np.zeros(self.sensor_number)
        self.dissonance0 = np.zeros(self.sensor_number)
        self.vacuity1 = np.zeros(self.sensor_number)
        self.dissonance1 = np.zeros(self.sensor_number)
        self.bl = np.ones(self.sensor_number) #initialized to 1
        self.fr = np.ones(self.sensor_number) #initialized to 1
        self.utility = None
        self.utility_rank = None
        self.mean_vacuity = None
        self.mean_dissonance = None
        self.mean_freshness = None
        self.mean_bl = None
        self.pre_utility = None

        self.initial_mq = np.zeros((4,),dtype=float)
        self.mq = None

        self.sn = None

        self.old_adj = None # the adjacency matrix of old sensor network

        self.adj_threshold = 0
        
        self.updates = 0

    def reset(self):
        self.database = np.full((self.sensor_number,6),-1, dtype=float)
        #animal ID,[sender ID,HR,AT,M,BL,T]
        self.total_database = np.zeros((self.sensor_number,4),dtype=float)#animal ID,[step,temperature,moving activity,heart rate]
        #self.total_database[:,3]=np.inf
        self.new_database = np.full((self.sensor_number,5),-1,dtype=float) #temp, hb, mv, bl

        # 1 - below normal, 2 - normal, 3 - above normal
        s = (self.sensor_number,3)
        self.evidence = np.ones(s)
        self.belief = np.zeros(s)
        self.uncertainty = np.zeros(self.sensor_number)
        self.vacuity0 = np.zeros(self.sensor_number)
        self.dissonance0 = np.zeros(self.sensor_number)
        self.vacuity1 = np.zeros(self.sensor_number)
        self.dissonance1 = np.zeros(self.sensor_number)
        self.bl = np.ones(self.sensor_number) #initialized to 1
        self.fr = np.ones(self.sensor_number) #initialized to 1
        self.utility = None
        self.utility_rank = None
        self.mean_vacuity = None
        self.mean_dissonance = None
        self.mean_freshness = None
        self.mean_bl = None
        self.pre_utility = 0

        self.initial_mq = np.zeros((4,),dtype=float)
        self.mq = deepcopy(self.initial_mq)

        self.sn = None

        self.old_adj = None # the adjacency matrix of old sensor network

        self.adj_threshold = 0
        
        self.ge = None # sensor network embedding

        self.updates = 0

# multi-agent world
class World(object):
    def __init__(self, T_m, alpha, um, pida, peda, pnca, nd):

        #create database
        self.database = []

        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []

        self.T_m = T_m
        self.alpha = alpha

        self.T_b = 0.1 #basic battery level to send information

        self.max_length = math.sqrt(161874)/2 #side length(40 acres to m^2)

        # communication channel dimensionality
        self.dim_c = 4
        self.total_k = nd # k=1,2,......,20
        
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        #self.dim_color = 3
        # sensors' update interval
        self.t_u = 30
        # agents' update interval
        self.t_a = 60
        # freshness decay rate
        self.lambd = 0.01
        self.steps = 0
        # uncertainty maximization indicator
        self.um = um
        # consolidated_rank
        self.consolidated_rank0 = None
        self.consolidated_rank1 = None
        #inside attackers
        self.inside_attackers = []        
        #outside attacker targets
        self.oa_targets = []
        # initial attack probability
        self.pida =  pida
        self.peda =  peda
        self.pnca =  pnca

        self.filter_count = 0

        self.total_count = 1
        
        self.overload = 0

        self.hb_list = [] # high battery level sensor list
        self.lb_list = [] # low battery level sensor list
        
        self.k_list = [] # debug tool
        
        self.rng1 = random.Random(0)
        self.rng2 = np.random.default_rng(0)
    
    def reset(self):
        self.steps = 0
        for sensor in self.landmarks:
            sensor.reset()
        for agent in self.agents:
            agent.reset()
            self.update_graph(agent)

    def set_ia(self):
        self.inside_attackers = self.rng1.sample(list(range(self.total_k)),int(self.total_k*self.pida))
        self.non_attackers = list(set(list(range(self.total_k)))-set(self.inside_attackers))

    def set_oa_targets(self):
        temp_list = []
        for i in self.non_attackers:
            if self.landmarks[i].bl > self.T_m:
                temp_list.append(i)
        self.oa_targets = temp_list#self.rng1.sample(temp_list,int(len(temp_list)*self.peda))

    # update state of the world
    def step(self):
        #flag = True
        self.rng1 = random.Random(self.steps)
        self.rng2 = np.random.default_rng(self.steps)
        # update the true info of sensors
        self.update_landmark_states()
        # update the actions of gateways(agents) (T_a = 60 s)
        self.get_blist()
        if self.steps%2 == 0:
            #self.total_count *=  len(lb_list)+1
            #print(len(lb_list),math.log(self.total_count,10)) 
            for agent in self.agents:
                self.update_graph(agent)
                #self.update_agent_state(agent)
                self.update_and_rank(agent)
            self.consolidate_rank()
        
        sparse0, sparse1 = [], []
        for s1 in self.hb_list:
            sparse0.append([])
            sparse1.append([])
            for s2 in self.lb_list:
                if distance_sensors[self.steps,s1.ID,s2.ID]<100:
                    if s2.ID in self.consolidated_rank0:
                        sparse0[-1].append(1)
                    else:
                        sparse0[-1].append(0)
                    if s2.ID in self.consolidated_rank1:
                        sparse1[-1].append(1)
                    else:
                        sparse1[-1].append(0)
                else:
                    sparse0[-1].append(0)
                    sparse1[-1].append(0)       
        graph0 = csr_matrix(sparse0)
        graph1 = csr_matrix(sparse1)
        match_list0 = maximum_bipartite_matching(graph0, perm_type='column')
        match_list1 = maximum_bipartite_matching(graph1, perm_type='column')
        ol0 = (len(self.consolidated_rank0)-sum(match_list0>=0))/(len(self.consolidated_rank0)+1e-8)
        ol1 = (len(self.consolidated_rank1)-sum(match_list1>=0))/(len(self.consolidated_rank1)+1e-8)
        if ol0 < ol1:
            match_list = match_list0
            self.overload = ol0
        else:
            match_list = match_list1
            self.overload = ol1
        
        self.rng1.shuffle(self.hb_list)
        self.rng1.shuffle(self.lb_list)
        #print(len(self.hb_list),len(self.lb_list))
        hb_idx = 0
        for s1 in self.hb_list:
            if s1.ID >= 5 and self.steps < 1122:
                s1.bl -= 85/5000000
            else:
                s1.bl -= 170/5000000 # gateway broadcast power consumption
            if s1.bl > self.T_m and match_list[hb_idx] > 0:
                #print("low_bl_list",low_bl_list)
                s3 = self.lb_list[match_list[hb_idx]]# select based on rank
                #print(s1.ID, s1.bl, s3.ID, s3.bl)
                s3.bl -= 0.145/5000000 #11*(27/2048)~=0.145
                # sensor broadcast power consumption
                if s1.ID >= 5 and self.steps < 1122:
                    s1.bl -= 85/5000000
                    if s3.ID in self.inside_attackers and self.rng1.random() < self.pida:
                        temp_data = self.rng1.choice(self.landmarks).data
                    else:
                        temp_data = s3.data
                        temp_data[4] = s3.bl
                    idx_order = [0,1]
                    random.shuffle(idx_order)
                    if range_agents[self.steps,s1.ID,self.agents[idx_order[0]].ID]:
                        self.update_gateway_data(self.agents[idx_order[0]], s3.ID, deepcopy(temp_data))
                else:
                    s1.bl -= 170/5000000
                    if s3.ID in self.inside_attackers and self.rng1.random() < self.pida:
                        temp_data = self.rng1.choice(self.landmarks).data
                    else:
                        temp_data = s3.data
                        temp_data[4] = s3.bl
                    for agent in self.agents:
                        if range_agents[self.steps,s1.ID,agent.ID]:
                            self.update_gateway_data(agent, s3.ID, deepcopy(temp_data))
            if s1.ID >= 5 and self.steps < 1122:
                if (s1.ID in self.inside_attackers and self.rng1.random() < self.pida) or (s1.ID in self.oa_targets and self.rng1.random() < self.peda):
                    temp_data = self.rng1.choice(self.landmarks).data
                else:
                    temp_data = s1.data
                    temp_data[4] = s1.bl
                idx_order = [0,1]
                random.shuffle(idx_order)
                if range_agents[self.steps,s1.ID,self.agents[idx_order[0]].ID]:
                    self.update_gateway_data(self.agents[idx_order[0]], s1.ID, deepcopy(temp_data))
            else:
                if (s1.ID in self.inside_attackers and self.rng1.random() < self.pida) or (s1.ID in self.oa_targets and self.rng1.random() < self.peda):
                    temp_data = self.rng1.choice(self.landmarks).data
                else:
                    temp_data = s1.data
                    temp_data[4] = s1.bl
                for agent in self.agents:
                    if range_agents[self.steps,s1.ID,agent.ID]:
                        self.update_gateway_data(agent, s1.ID, deepcopy(temp_data))
            hb_idx += 1
        self.steps += 1
        #return flag

    def update_graph(self,agent): # update the sensor network
        sn = nx.Graph() # initializa sensor network
        num_sensor = len(self.landmarks)
        for i in range(num_sensor):
             sn.add_node(i,feature=0)
        self.get_agent_blist(agent)
        #print(len(agent.hb_list),len(agent.lb_list))
        for s1 in agent.hb_list:
            for s2 in agent.lb_list:
                #print(self.steps,s1,s2,distance_sensors[self.steps,s1,s2])
                if distance_sensors[self.steps,s1,s2]<100:
                    sn.add_edge(s1, s2) # add edges between sensors within wireless range
                    #print(self.steps,s1,s2,distance_sensors[self.steps,s1,s2])
                #else:
                    #print(self.steps,s1,s2,distance_sensors[self.steps,s1,s2])
        agent.sn = sn
        #print(agent.ID,np.sum(nx.adjacency_matrix(agent.sn)))

    def check_graph(self,agent):
        # compare adjacency matrix
        adj = nx.adjacency_matrix(agent.sn)
        if agent.old_adj == None:
            agent.old_adj = deepcopy(adj)
            return True
        elif np.sum(abs(agent.old_adj - adj)) > agent.adj_threshold:
            #print(agent.ID, np.sum(abs(agent.old_adj - adj)),np.sum(agent.old_adj),np.sum(adj))
            agent.old_adj = deepcopy(adj)
            return True
        else:
            #print(agent.ID, np.sum(abs(agent.old_adj - adj)),np.sum(agent.old_adj),np.sum(adj))
            return False

    def get_blist(self):
        #self.set_ia()
        self.set_oa_targets()
        lb_list, hb_list = [], []
        for sensor in self.landmarks:
            if sensor.bl > self.T_m:
                if sensor.ID in self.inside_attackers and self.rng1.random() < self.pnca:
                    if self.rng1.random() < self.pida:
                        lb_list.append(sensor)
                else:
                    hb_list.append(sensor)
            else:
                #print(sensor.bl)
                lb_list.append(sensor)
        self.lb_list = lb_list
        self.hb_list = hb_list
        #print("hb:%d,lb:%d"%(len(self.hb_list),len(self.lb_list)))

    def get_agent_blist(self,agent):
        lb_list, hb_list = [], []
        for sensor in self.landmarks:
            if range_agents[self.steps,sensor.ID,agent.ID]:
                if sensor.bl > self.T_m:
                    hb_list.append(sensor.ID)
                elif sensor.bl > self.T_b :
                    lb_list.append(sensor.ID)
        agent.lb_list = lb_list
        agent.hb_list = hb_list
        #print("agent:%d,hb:%d,lb:%d"%(agent.ID,len(agent.hb_list),len(agent.lb_list)))
        #print(agent.bl)

    def update_landmark_states(self): # update sensors' states (vac, diss, fr, bl)
        #self.update_landmark_position()
        for landmark in self.landmarks:
            self.update_bl(landmark)
            landmark.temp = self.rng2.normal(38,1,1)[0]
            landmark.t = self.t_u*self.steps
            landmark.data = [landmark.ID, landmark.temp, hbs[self.steps,landmark.ID], vels[self.steps,landmark.ID], landmark.bl, landmark.t]        
            #print(landmark.bl)

    def update_bl(self, landmark):
        # charge
        t_hour = self.steps*self.t_u/3600
        if t_hour<24:
            center = 12
        else:
            center = 36
        prob = self.alpha*max(0,-1/6*pow(t_hour-0.1/self.max_length*(locs[self.steps,landmark.ID,0]-self.max_length)-center,2)+1)
        if self.rng1.uniform(0,1)<prob and landmark.bl < 1: #charge battery
            landmark.bl += min(1-landmark.bl, 0.00004*self.t_u)# 1/((5*1000000)/200)=1/25000=0.00004

        # consume
        if landmark.bl < self.T_m:
            if landmark.bl > 0:
                landmark.bl -= 0.00056*self.t_u/5000000 # 2/(60*60)~=0.00056
        else:
            landmark.bl -= 0.0022*self.t_u/5000000 # 8/(60*60)~=0.0022

    def update_and_rank(self, agent):
        # get the indices where agent.bl=-1
        indices = np.where(agent.bl>0)
        #print("vacuity:",agent.vacuity,"dissonance:",agent.dissonance,"fr:",agent.fr,"bl:",agent.bl)
        agent.utility0 = 2-agent.vacuity0-agent.dissonance0+agent.fr-pow(agent.bl-self.T_m,2)
        agent.utility1 = 2-agent.vacuity1-agent.dissonance1+agent.fr-pow(agent.bl-self.T_m,2)
        #needs to update (use sensor network connectivity filter)
        temp_list = []
        cc = []
        for i in range(len(self.landmarks)):
            if agent.sn.degree[i]>0 and i in agent.lb_list:
                temp_list.append(i)
        for i in range(len(self.landmarks)):
            if agent.sn.degree[i]>0:
                cc.append(i)
        filtered_list0 = deepcopy(agent.utility0)[temp_list] #filtered values
        filtered_list1 = deepcopy(agent.utility1)[temp_list] #filtered values
        if len(filtered_list0) == 0 or agent.action == 0:
            agent.utility_rank0 = agent.utility_rank1 = []
        else:
            #ranked idx
            temp_list = np.array(temp_list)
            if agent.action == 1:
                temp_rank0 = np.argsort(filtered_list0)
                base_list0 = temp_list[temp_rank0]
                temp_rank1 = np.argsort(filtered_list1)
                base_list1 = temp_list[temp_rank1]
                agent.utility_rank0 = base_list0[:int(len(base_list0)/2)]
                agent.utility_rank1 = base_list1[:int(len(base_list1)/2)]
            else:
                agent.utility_rank0 =  agent.utility_rank1 = temp_list

    def consolidate_rank(self):
        temp_rank0 = np.arange(self.total_k,dtype=float)
        temp_rank1 = np.arange(self.total_k,dtype=float)
        for agent in self.agents:
            temp_rank0 = np.intersect1d(temp_rank0, agent.utility_rank0)
            temp_rank1 = np.intersect1d(temp_rank1, agent.utility_rank1)
        self.consolidated_rank0 = deepcopy(temp_rank0)
        self.consolidated_rank1 = deepcopy(temp_rank1)


    def update_gateway_data(self, agent, sender_id, data):
        self.calculate_opinion(agent, data)
        agent.new_database[data[0]][0] = data[1] # temp
        agent.new_database[data[0]][1] = data[2] # hb
        agent.new_database[data[0]][2] = data[3] # mv
        agent.new_database[data[0]][3] = data[4] # bl
        agent.new_database[data[0]][4] = data[5] # t
        agent.total_database[data[0]][0] += 1
        agent.total_database[data[0]][1] += data[1]
        agent.total_database[data[0]][2] += data[3]
        agent.total_database[data[0]][3] += data[2]
        agent.database[data[0]] = [sender_id, agent.total_database[data[0]][3]/agent.total_database[data[0]][0],
                                            agent.total_database[data[0]][1]/agent.total_database[data[0]][0],
                                            agent.total_database[data[0]][2]/agent.total_database[data[0]][0],
                                            data[4],
                                            data[5]
                                            ]
        agent.fr = np.exp(self.lambd*(agent.database[:,5]-self.steps*self.t_u))
        agent.bl = agent.database[:,4]
        agent.updates += 1
        
    def calculate_opinion(self, agent, data):
        #animal ID, temperature, heart beat, moving activity, battery level, time stamp
        id = data[0]
        temp = data[1]
        hb = data[2]
        ma = data[3]
        bl = data[4]
        time_stamp = data[5]

        base_rate = 1/3
        W = 3

        if temp>39.2:
            agent.evidence[id, 2] += 1
        elif temp<37.8: 
            agent.evidence[id, 0] += 1
        else:
            agent.evidence[id, 1] += 1

        if hb>84:
            agent.evidence[id, 2] += 1
        elif hb<48: 
            agent.evidence[id, 0] += 1
        else:
            agent.evidence[id, 1] += 1

        if  ma>2:
            agent.evidence[id, 2] += 1
        elif ma<1: 
            agent.evidence[id, 0] += 1
        else:
            agent.evidence[id, 1] += 1

        agent.belief[id, 0] = agent.evidence[id, 0] / np.sum(agent.evidence[id])
        agent.belief[id, 1] = agent.evidence[id, 1] / np.sum(agent.evidence[id])
        agent.belief[id, 2] = agent.evidence[id, 2] / np.sum(agent.evidence[id])


        agent.uncertainty[id] = W / np.sum(agent.evidence[id])
        agent.vacuity0[id] = agent.uncertainty[id]

        b = agent.belief[id,:]
        diss = 0
        for b1 in b: 
            temp_sum = 0
            denom_sum = 0
            for b2 in b:
                if b1!=b2:
                    bal = 1 - (abs(b2-b1)/(b1+b2))
                    temp_sum = temp_sum + b2*bal
                    denom_sum = denom_sum + b2

            #print(diss,b1*temp_sum,denom_sum)
            #assert diss==0
            diss = diss + (b1*temp_sum)/(denom_sum+1e-8)
      
        agent.dissonance0[id] = diss
        
        if self.um and agent.uncertainty[id] <= 0.05: #uncertainty maximization
            P = np.zeros(3)
            P[0] = agent.belief[id, 0] + base_rate * agent.uncertainty[id]
            P[1] = agent.belief[id, 1] + base_rate * agent.uncertainty[id]
            P[2] = agent.belief[id, 2] + base_rate * agent.uncertainty[id]

            P_max = P/base_rate
            agent.uncertainty[id] = np.amin(P_max)
            agent.vacuity1[id] = agent.uncertainty[id]

            # Updated belief 
            agent.belief[id, 0] = P[0] - (base_rate*agent.uncertainty[id])
            agent.belief[id, 1] = P[1] - (base_rate*agent.uncertainty[id])
            agent.belief[id, 2] = P[2] - (base_rate*agent.uncertainty[id])
            
            b = agent.belief[id,:]
            diss = 0
            for b1 in b: 
                temp_sum = 0
                denom_sum = 0
                for b2 in b:
                    if b1!=b2:
                        bal = 1 - (abs(b2-b1)/(b1+b2))
                        temp_sum = temp_sum + b2*bal
                        denom_sum = denom_sum + b2

                #print(diss,b1*temp_sum,denom_sum)
                #assert diss==0
                diss = diss + (b1*temp_sum)/(denom_sum+1e-8)
          
            agent.dissonance1[id] = diss
        else:
            agent.vacuity1[id] = agent.vacuity0[id]
            agent.dissonance1[id] = agent.dissonance0[id]

class Scenario():
    def __init__(self,arglist):
        self.max_length = math.sqrt(161874)/2
        self.arglist = arglist
        self.num_landmarks = arglist.nd#20
        self.num_agents = arglist.m # initial 2
        self.w0,self.w1,self.w2,self.w3 = 1,1,1,1
        self.T_m = 0.3
        self.lamda = 0.9
        self.margins = [4.24,36,0.424,1]#3*\sqrt(2)*\sigma

    def make_world(self):
        world = World(T_m=0.3, alpha=self.arglist.alpha, um=self.arglist.um, pida = self.arglist.pida, peda = self.arglist.peda, pnca = self.arglist.pnca, nd = self.arglist.nd)
        # add landmarks（cows）
        world.landmarks = [Landmark(i,self.arglist.tl) for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
        # add agents
        world.agents = [Agent(i,len(world.landmarks)) for i in range(self.num_agents)]
        world.database = np.full((len(world.landmarks),5),-1, dtype=float)
        #[temp,hb,mv,bl]

        s = (len(world.landmarks),3)
        world.evidence = np.ones(s)
        world.belief = np.zeros(s)
        world.uncertainty = np.zeros(len(world.landmarks))
        world.vacuity = np.zeros(len(world.landmarks))
        world.dissonance = np.zeros(len(world.landmarks))
        #world.agents[0].pos = np.array([-self.max_length/2,0])#np.zeros(world.dim_p)
        #world.agents[1].pos = np.array([self.max_length/2,0])#np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, initial_steps=0):
        world.set_ia()
        # random properties for agents
        world.reset()

    def reward(self, agent, world):
        #need to be checked
        for i in range(self.num_landmarks):
            idx = np.argmax([world.agents[j].new_database[i][4] for j in range(self.num_agents)])
            world.database[i] = world.agents[idx].new_database[i]
        count = 0
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps-1,i],vels[world.steps-1,i],landmark.bl]
            for x,y in zip(world.database[i],attributes):
                if x!=y:
                    count += 1
        #print(count)
        agent.pre_utility += -count/(self.num_landmarks*4)-world.overload
        return agent.pre_utility


    def monitoring_quality_global(self, world):
        # database: [sender ID,HR,AT,MXT,MNT,M,BL,T,D]
        count = 0
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps-1,i],vels[world.steps-1,i],landmark.bl]# use actual value
            for x,y in zip(world.database[i],attributes):
                if x!=y:
                    count += 1
        #print(count)
        return count

    def vacuity(self, world):
        indices = np.where(world.database[:,4]>0)
        mv = np.mean(world.vacuity[indices])
        return mv
    
    def dissonance(self, world):
        indices = np.where(world.database[:,4]>0)
        md = np.mean(world.dissonance[indices])
        return md
    
    def freshness(self, world):
        fr = np.exp(world.lambd*(world.database[:,4]-world.steps*world.t_u))
        return fr

    def energy_diff(self,world):

        indices = np.where(world.database[:,4]>0)
        mbl = np.mean(world.database[:,3][indices])
        return -self.w3*math.pow(mbl-self.T_m,2)

    def metrics(self,world):
        
        mq = self.monitoring_quality_global(world)
        #if count > 0:
            #print(count)
        vc = self.vacuity(world) 
        do = self.dissonance(world)
        fs = self.freshness(world)
        er = self.energy_diff(world)
        ol = world.overload
        return mq, vc, do, fs, er, ol

class ENV_test(object):
    def __init__(self, length, scenario, nagents):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world()
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions=[2,2]):
        done = False
        temp_metric = []
        for i in range(self.nagents):
            self.world.agents[i].action = actions[i]
        for i in range(self.nagents):
            self.world.agents[i].pre_utility = 0
        while self.world.steps < (1+self.steps)*self.interval:
            self.world.step()
            self.world.step()
            temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
            temp_metric.append(self.sf.metrics(self.world))
        if self.steps == self.length-1:
            done = True
            while self.world.steps < 2878:
                self.world.step()
                self.world.step()
                temp_metric.append(self.sf.metrics(self.world))
        for i in range(self.nagents):
            self.state[i,self.steps] = self.world.agents[i].updates
            self.world.agents[i].updates = 0
        self.steps += 1
        if done:
            r = temp_r
        else:
            r = temp_r
        return self.state, r, done, temp_metric

arglist = parse_args()
if arglist.m!=2:
    pre_dir = "env/m/%d/"%(arglist.m)
else:
    pre_dir = "env/n/%d/"%(arglist.nd)
locs = np.load(pre_dir+"locs.npy")
vels = np.load(pre_dir+"vels.npy")
hbs = np.load(pre_dir+"hbs.npy")
distance_sensors = np.load(pre_dir+"distance_sensors.npy")
range_agents = np.load(pre_dir+"range_agents.npy")

if __name__ == "__main__":
    rewards_list = []
    metrics_list = []
    final_reward_list = []

    sf = Scenario(arglist)
    length = 10
    nagents = 2
    env = ENV_test(length, sf, nagents) 
    for i in range(1,101):
        state = env.reset()
        final_reward = 0
        metric_list = []
        for t in range(100):        
            # select action with policy
            #actions = [random.randint(0,2),random.randint(0,2)]
            actions = [random.randint(0,2)]*2
            state, reward, done, temp_metric = env.step(actions)
            final_reward += sum(reward)/2
            metric_list += temp_metric
            if done:
                break
        metrics_list.append(metric_list)
        final_reward_list.append(final_reward)
        print("Progress:%d/%d"%(i,100))
    directory = "results_dm_new/dm_%.1f_%.1f_%.1f_%.2f_%d_%.1f/"%(arglist.pida,arglist.peda,arglist.pnca,arglist.tl,arglist.nd,arglist.alpha)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(metrics_list,open(directory+"metrics_list.pkl","wb"))
    pickle.dump(final_reward_list,open(directory+"final_reward_list.pkl","wb"))
    #os.system('say "you program is finishied"')
