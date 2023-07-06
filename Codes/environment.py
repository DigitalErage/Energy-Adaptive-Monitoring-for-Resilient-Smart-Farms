import math
import random
import numpy as np
from core import World, Agent, Landmark
from copy import deepcopy
import pickle
#from multiprocessing import shared_memory
import shared_memory

locs = np.load("env/locs.npy")
vels = np.load("env/vels.npy")
hbs = np.load("env/hbs.npy")
distance_sensors = np.load("env/distance_sensors.npy")
range_agents = np.load("env/range_agents.npy")
#intervals = pickle.load(open("intervals.pkl","rb"))

class Scenario():
    def __init__(self):
        self.max_length = math.sqrt(161874)/2
        self.num_landmarks = 20#20
        self.num_agents = 2
        self.w0,self.w1,self.w2,self.w3 = 1,1,1,1
        self.T_m = 0.3
        self.lamda = 0.9
        self.margins = [4.24,36,0.424,1]#3*\sqrt(2)*\sigma

    def make_world(self, um, ap=0.3):
        world = World(um=um,ap=ap)
        # add landmarks（cows）
        world.landmarks = [Landmark(i) for i in range(self.num_landmarks)]
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
        diff_quality = np.zeros((4,),dtype=float)
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps,i],vels[world.steps,i],landmark.bl]# use actual value
            for idx in range(4):
                diff_quality[idx] += min(1,abs(agent.new_database[i][idx]-attributes[idx])/self.margins[idx])
        agent.pre_utility += -np.sum(diff_quality/(self.num_landmarks*4))-world.overload
        return agent.pre_utility

    def monitoring_quality(self, agent, world):
        # database: [sender ID,HR,AT,MXT,MNT,M,BL,T,D]
        diff_quality = np.zeros((4,),dtype=float)
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps,i],vels[world.steps,i],landmark.bl]# use actual value
            for idx in range(4):
                diff_quality[idx] += min(1,abs(agent.new_database[i][idx]-attributes[idx])/self.margins[idx])
        agent.mq = np.sum(diff_quality/(self.num_landmarks*4))#(self.lamda*agent.mq+diff_quality)/(1+self.lamda)
        return agent.mq
        #return self.mq/(self.num_landmarks*4)

    def monitoring_quality_global(self, world):
        # database: [sender ID,HR,AT,MXT,MNT,M,BL,T,D]
        diff_quality = np.zeros((4,),dtype=float)
        count1,count2 = 0,0
        for i, landmark in enumerate(world.landmarks):
            attributes = [landmark.temp,hbs[world.steps-1,i],vels[world.steps-1,i],landmark.bl]# use actual value
            for idx in range(4):
                diff_quality[idx] += min(1,abs(world.database[i][idx]-attributes[idx])/self.margins[idx])
            for x,y in zip(world.database[i],attributes):
                if x==y:
                    count1 += 1
                    #print("(data consistence step,sensor id):(%d,%d)" % (world.steps,i))
                else:
                    count2 +=1
                    #print(world.database[i][4]/30,world.steps,i,x,y)
        # if count != 0:
        #     print('count: ' + str(count))
        #print('total: ' + str(total))
        #print(count/total)
        mq = np.sum(diff_quality/(self.num_landmarks*4))#(self.lamda*agent.mq+diff_quality)/(1+self.lamda)
        return mq, count1, count2

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

        agentA = world.agents[0]
        agentB = world.agents[1]

        #need to be checked
        index = 0
        for a,b in zip(agentA.new_database,agentB.new_database):
            if a[4] > b[4]:
                world.database[index] = a
            else:
                world.database[index] = b
            index+=1
        
        mq, count1, count2 = self.monitoring_quality_global(world)
        #if count > 0:
            #print(count)
        vc = self.vacuity(world) 
        do = self.dissonance(world)
        fs = self.freshness(world)
        er = self.energy_diff(world)
        ol = world.overload
        return mq, vc, do, fs, er, ol, count1, count2

class ENV_dqn(object):
    def __init__(self, length, scenario, nagents, um = False, ap = 0.3):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)#np.array([0,1,2,3],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world(um=um,ap=ap)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions,eval_set1 = set(),eval_set2 = set()):
        done = False
        temp_metric = []
        for i in range(self.nagents):
            #self.state[i,self.steps] = actions[i] + 1
            self.world.agents[0].action = self.world.agents[1].action = actions[i]
        # advance world state
        #for i in range(self.nagents):
        self.world.agents[0].pre_utility = self.world.agents[1].pre_utility = 0
        while self.world.steps < (1+self.steps)*self.interval:
            self.world.step()
            self.world.step()
            temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(2)]
            temp_metric.append([self.sf.metrics(self.world.agents[i], self.world) for i in range(2)])
        if self.steps == self.length-1:
            done = True
            while self.world.steps < 2878:
                self.world.step()
                self.world.step()
                temp_metric.append([self.sf.metrics(self.world) for i in range(2)])
        for i in range(self.nagents):
            self.state[i,self.steps] = self.world.agents[i].updates
            self.world.agents[i].updates = 0
        self.steps += 1
        #print("DRL:",self.world.steps)
        #print(self.area)
        #temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
        if done:
            #print(self.state)
            eval_set1.add(sum([self.state[0,i] for i in range(self.length)]))
            #eval_set1.add(sum([self.state[0,i]*pow(3,i) for i in range(self.length)]))
            #eval_set2.add(sum([self.state[1,i]*pow(3,i) for i in range(self.length)]))
            r = sum(temp_r)
            #print(r)
        else:
            r = sum(temp_r)#r = [0 for i in range(self.nagents)]
        return self.state, r, done, eval_set1, eval_set2, temp_metric
    
class ENV_ppo(object):
    def __init__(self, length, scenario, nagents, um = False, ap = 0.3):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)#np.array([0,1,2,3],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world(um=um,ap=ap)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions,eval_set1 = set(),eval_set2 = set()):
        done = False
        temp_metric = []
        for i in range(self.nagents):
            #self.state[i,self.steps] = actions[i] + 1
            self.world.agents[i].action = actions[i]
        # advance world state
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
                #print('metri: ')
                #print(self.sf.metrics(self.world)[6])
        for i in range(self.nagents):
            self.state[i,self.steps] = self.world.agents[i].updates
            self.world.agents[i].updates = 0
        self.steps += 1
        #print("DRL:",self.world.steps)
        #print(self.area)
        #temp_r = [self.sf.reward(self.world.agents[i], self.world) for i in range(self.nagents)]
        if done:
            #print(self.state)
            eval_set1.add(sum([self.state[0,i] for i in range(self.length)]))
            #eval_set1.add(sum([self.state[0,i]*pow(3,i) for i in range(self.length)]))
            eval_set2.add(sum([self.state[1,i]*pow(3,i) for i in range(self.length)]))
            r = temp_r
            #print(r)
        else:
            r = temp_r#r = [0 for i in range(self.nagents)]
        return self.state, r, done, eval_set1, eval_set2, temp_metric

class ENV_test(object):
    def __init__(self, length, scenario, nagents, um = False):
        self.length = length
        self.interval = int(1166/self.length)
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)
        self.sf = scenario
        self.world = self.sf.make_world(um=um)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.world.set_ia()
        self.world.reset()
        return self.state
    
    def step(self,actions):
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
        if self.steps == self.length-1:
            done = True
            while self.world.steps < 2878:
                self.world.step()
                self.world.step()
        for i in range(self.nagents):
            self.state[i,self.steps] = self.world.agents[i].updates
            self.world.agents[i].updates = 0
        self.steps += 1
        if done:
            r = temp_r
        else:
            r = temp_r
        return self.state, r, done

class ENV_tree1(object):
    def __init__(self, length, nagents):
        self.length = length
        self.nagents = nagents
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        self.steps = 0
        self.action_space = np.array([0,1,2],dtype=int)
        
    def reset(self):
        self.steps = 0
        self.state = np.zeros((self.nagents,self.length),dtype=float)
        return self.state
    
    def step(self,actions):
        done = False
        for i in range(self.nagents):
            self.state[i,self.steps] = actions[i] + 1
        total_action = 3*actions[0] + actions[1]
               
        if self.steps == self.length-1:
            done = True
        self.steps += 1
        if done:
            r = total_action#temp_r
        else:
            r = total_action#temp_r
        return self.state, r, done

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
