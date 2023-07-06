import numpy as np
import random
import math
from copy import deepcopy
import networkx as nx
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

locs = np.load("env/locs.npy")
vels = np.load("env/vels.npy")
hbs = np.load("env/hbs.npy")
distance_sensors = np.load("env/distance_sensors.npy")
range_agents = np.load("env/range_agents.npy")
T_m = 0.3
init_bl = 0.3
alpha = 0.8

# properties of landmark entities
class Landmark():
    def __init__(self,ID):
        super(Landmark, self).__init__()
        self.T_m = T_m
        self.ID = ID #animal ID
        self.temp_0 = np.random.normal(38,1,1)[0] #temperature
        self.temp = deepcopy(self.temp_0)
        self.hb = hbs[0,ID] #heart beat
        self.vel = vels[0,ID]
        if self.ID<5:#100:
            self.bl = 1#lower the battery level
        else:
            self.bl = init_bl+self.ID*170/5000000#self.ID*
        self.t = 0 #time stamp with sec. as unit
        self.data = [self.ID, self.temp, self.hb, self.vel, self.bl,self.t]
        
    def reset(self):
        self.temp = deepcopy(self.temp_0)
        self.hb = hbs[0,self.ID] #heart beat
        self.vel = vels[0,self.ID]
        if self.ID<5:#100:
            self.bl = 1#lower the battery level
        else:
            self.bl = init_bl+self.ID*170/5000000#self.ID*
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
        self.vacuity = np.zeros(self.sensor_number)
        self.dissonance = np.zeros(self.sensor_number)
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

        self.ge = None # sensor network embedding

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
        self.vacuity = np.zeros(self.sensor_number)
        self.dissonance = np.zeros(self.sensor_number)
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
    def __init__(self, um=True,ap=0.3):

        #create database
        self.database = []

        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []

        self.T_m = T_m

        self.T_b = 0.1 #basic battery level to send information

        self.max_length = math.sqrt(161874)/2 #side length(40 acres to m^2)

        # communication channel dimensionality
        self.dim_c = 4
        self.total_k = 20 # k=1,2,......,20
        
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
        self.consolidated_rank = None
        #inside attackers
        self.inside_attackers = []        
        #outside attacker targets
        self.oa_targets = []
        # initial attack probability
        self.iap = self.ap = ap#0.3   
        # attack probability
        #self.ap = 0#0.5

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
            #agent.ge = self.get_embedding(agent)

    def set_ia(self):
        self.inside_attackers = self.rng1.sample(list(range(self.total_k)),int(self.total_k*self.iap))
        self.non_attackers = list(set(list(range(self.total_k)))-set(self.inside_attackers))

    def set_oa_targets(self):
        temp_list = []
        for i in self.non_attackers:
            if self.landmarks[i].bl > self.T_m:
                temp_list.append(i)
        self.oa_targets = self.rng1.sample(temp_list,int(len(temp_list)*self.iap))

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
                #if self.check_graph(agent):
                    #flag = False
                    #agent.ge = self.get_embedding(agent)
                #print(agent.utility_rank)
            self.consolidate_rank()

        self.rng1.shuffle(self.hb_list)
        self.rng1.shuffle(self.lb_list)
        #print(len(self.hb_list),len(self.lb_list))
        
        sparse = []
        for s1 in self.hb_list:
            sparse.append([])
            for s2 in self.lb_list:
                if s2.ID in self.consolidated_rank and distance_sensors[self.steps,s1.ID,s2.ID]<100:
                    sparse[-1].append(1)
                else:
                    sparse[-1].append(0)            
        graph = csr_matrix(sparse)
        match_list = maximum_bipartite_matching(graph, perm_type='column')
        if self.steps%2 == 0:
            self.overload = (len(self.consolidated_rank) - sum(match_list>=0))/(len(self.consolidated_rank)+1e-8)
        
        hb_idx = 0
        for s1 in self.hb_list:
            s1.bl -= 170/5000000 # gateway broadcast power consumption
            if s1.bl > self.T_m and match_list[hb_idx] > 0:
                #print("low_bl_list",low_bl_list)
                s3 = self.lb_list[match_list[hb_idx]]# select based on rank
                #print(s1.ID, s1.bl, s3.ID, s3.bl)
                s3.bl -= 0.145/5000000 #11*(27/2048)~=0.145
                # sensor broadcast power consumption
                s1.bl -= 170/5000000 # gateway broadcast power consumption
                # consume additional power? double the energy
                if s3.ID in self.inside_attackers and self.rng1.random() < self.ap:
                    temp_data = self.rng1.choice(self.landmarks).data
                else:
                    temp_data = s3.data
                    temp_data[4] = s3.bl
                for agent in self.agents:
                    if range_agents[self.steps,s1.ID,agent.ID]:
                        self.update_gateway_data(agent, s3.ID, deepcopy(temp_data))
            if (s1.ID in self.inside_attackers or s1.ID in self.oa_targets) and self.rng1.random() < self.ap:
                temp_data = self.rng1.choice(self.landmarks).data
                #print("(data modified step,sensor id):(%d,%d)" % (self.steps,s1.ID))
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

    def get_embedding(self,agent): # use graph2vec embedding the sensor network
        g2v = Graph2Vec(wl_iterations= 2, attributed= True, dimensions= 32, workers = 10, 
             down_sampling = 0.0001, epochs = 10, learning_rate= 0.025, min_count = 5,
             seed = 42, erase_base_features = False)
        g2v.fit([agent.sn])
        ge = g2v.get_embedding()
        return ge[0]

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
        self.set_ia()
        self.set_oa_targets()
        lb_list, hb_list = [], []
        for sensor in self.landmarks:
            if sensor.bl > self.T_m:
                if sensor.ID in self.inside_attackers and self.rng1.random() < self.ap:
                    if self.rng1.random() < self.ap:
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
        prob = alpha*max(0,-1/6*pow(t_hour-0.1/self.max_length*(locs[self.steps,landmark.ID,0]-self.max_length)-center,2)+1)
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
        agent.mean_vacuity = np.mean(agent.vacuity[indices])
        agent.mean_dissonance = np.mean(agent.dissonance[indices])
        agent.mean_freshness = np.mean(agent.fr[indices])
        agent.mean_bl = np.mean(agent.bl[indices])
        #print("vacuity:",agent.vacuity,"dissonance:",agent.dissonance,"fr:",agent.fr,"bl:",agent.bl)
        agent.utility = 2-agent.vacuity-agent.dissonance+agent.fr-pow(agent.bl-self.T_m,2)
        #needs to update (use sensor network connectivity filter)
        temp_list = []
        cc = []
        for i in range(len(self.landmarks)):
            if agent.sn.degree[i]>0 and i in agent.lb_list:
                temp_list.append(i)
        for i in range(len(self.landmarks)):
            if agent.sn.degree[i]>0:
                cc.append(i)
        filtered_list = deepcopy(agent.utility)[temp_list] #filtered values
        temp_rank = np.argsort(filtered_list)
        if len(temp_rank) == 0 or agent.action == 0:
            agent.utility_rank = []
        else:
            #ranked idx
            temp_list = np.array(temp_list)
            base_list = temp_list[temp_rank]
            if agent.action == 1:
                agent.utility_rank = base_list[:int(len(base_list)/2)]
            else:
                agent.utility_rank = base_list

    def consolidate_rank(self):
        temp_rank = np.arange(self.total_k,dtype=float)
        for agent in self.agents:
            temp_rank = np.intersect1d(temp_rank, agent.utility_rank)
        self.consolidated_rank = deepcopy(temp_rank)

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
        agent.vacuity[id] = agent.uncertainty[id]
        
        if self.um and agent.uncertainty[id] <= 0.05: #uncertainty maximization
            P = np.zeros(3)
            P[0] = agent.belief[id, 0] + base_rate * agent.uncertainty[id]
            P[1] = agent.belief[id, 1] + base_rate * agent.uncertainty[id]
            P[2] = agent.belief[id, 2] + base_rate * agent.uncertainty[id]

            P_max = P/base_rate
            agent.uncertainty[id] = np.amin(P_max)
            agent.vacuity[id] = agent.uncertainty[id]

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
      
        agent.dissonance[id] = diss

    def calculate_opinion(self, world, data):
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
            world.evidence[id, 2] += 1
        elif temp<37.8: 
            world.evidence[id, 0] += 1
        else:
            world.evidence[id, 1] += 1

        if hb>84:
            world.evidence[id, 2] += 1
        elif hb<48: 
            world.evidence[id, 0] += 1
        else:
            world.evidence[id, 1] += 1

        if  ma>2:
            world.evidence[id, 2] += 1
        elif ma<1: 
            world.evidence[id, 0] += 1
        else:
            world.evidence[id, 1] += 1

        world.belief[id, 0] = world.evidence[id, 0] / np.sum(world.evidence[id])
        world.belief[id, 1] = world.evidence[id, 1] / np.sum(world.evidence[id])
        world.belief[id, 2] = world.evidence[id, 2] / np.sum(world.evidence[id])


        world.uncertainty[id] = W / np.sum(world.evidence[id])
        world.vacuity[id] = world.uncertainty[id]
        
        if self.um and world.uncertainty[id] <= 0.05: #uncertainty maximization
            P = np.zeros(3)
            P[0] = world.belief[id, 0] + base_rate * world.uncertainty[id]
            P[1] = world.belief[id, 1] + base_rate * world.uncertainty[id]
            P[2] = world.belief[id, 2] + base_rate * world.uncertainty[id]

            P_max = P/base_rate
            world.uncertainty[id] = np.amin(P_max)
            world.vacuity[id] = world.uncertainty[id]

            # Updated belief 
            world.belief[id, 0] = P[0] - (base_rate*world.uncertainty[id])
            world.belief[id, 1] = P[1] - (base_rate*world.uncertainty[id])
            world.belief[id, 2] = P[2] - (base_rate*world.uncertainty[id])
            
        b = world.belief[id,:]
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
      
        world.dissonance[id] = diss
