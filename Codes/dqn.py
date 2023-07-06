import torch
import torch.nn as nn
import torch.optim as optim
#from drl_env import ReplayBuffer
import random
import numpy as np


device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    #print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    pass

class RolloutBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.buffer = []
        self.batch_size = batch_size

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0, a, r, s1, done))

    def sample(self):
        #print(random.sample(self.buffer, batch_size))
        s0, a, r, s1, done = zip(*random.sample(self.buffer, self.batch_size))
        s0 = torch.tensor(np.array(s0), dtype=torch.float).to(device)
        s1 = torch.tensor(np.array(s1), dtype=torch.float).to(device)
        a = torch.tensor(np.array(a), dtype=torch.long).to(device)
        r = torch.tensor(np.array(r), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.float).to(device)
        return s0, a, r, s1, done

    def size(self):
        return len(self.buffer)

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, nagents, batch_size, gamma, lr):
        super(DQNAgent, self).__init__()
        self.batch_size = batch_size
        self.buffer = RolloutBuffer(batch_size,batch_size)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = lr
        size = 256
        #224:optimal:5/100,better than greedy:55/100, at least greedy:76/100,320.0211285299997s
        #280:optimal:12/100,better than greedy:64/100, at least greedy:86/100,434.8715323809997s

        self.nn = nn.Sequential(
            nn.Linear(self.state_size*nagents, size),
            nn.ReLU(),
            nn.Linear(size, 2*size),
            nn.ReLU(),
            #nn.Linear(2*size, 2*size),
            #nn.ReLU(),
            nn.Linear(2*size, size),
            nn.ReLU(),
            nn.Linear(size, self.action_size)
        ).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, amsgrad=True)#no need cuda
        self.is_training = True
        

    def model(self, x):
        return self.nn(x)
    
    def select_action(self, obs, idx):
        state = torch.FloatTensor(obs.flatten()).to(device)
        #state = torch.tensor(obs[idx], dtype=torch.float).to(device)
        #print(random.random())
        if random.random() > self.epsilon or not self.is_training:            
            q_value = self.model(state)
            #print(q_value.size())
            action = q_value.max(0)[1].item()
        else:
            action = random.randrange(self.action_size)
            #print(action)
        return action
    
    def remember(self, state, action, reward, next_state, done):#, batch_size):
        self.buffer.add(state, action, reward, next_state, done)
        #if self.memory.size() < batch_size:
            #self.memory.add(state, action, reward, next_state, done)

    def update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample()

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]
        
        #print(q_values.size(),a.unsqueeze(1).size())
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fgs(self, epsilon=0.2):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample()

        fs0 = s0.clone().requires_grad_().to(device)

        q_values = self.model(fs0)
        next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]
        
        #print(q_values.size(),a.unsqueeze(1).size())
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        #self.optimizer.zero_grad()
        loss.backward()
        sg_sign = torch.sign(fs0.grad.to(fs0.device)) 

        fs0 = s0 + epsilon * s0 * sg_sign

        fs0.detach_()

        q_values = self.model(fs0)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def bim(self, epsilon=0.2, steps=2, alpha=0.1):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample()

        fs0 = s0.clone()

        for i in range(steps):
            fs0.requires_grad = True
            q_values = self.model(fs0)
            next_q_values = self.model(s1)
            next_q_value = next_q_values.max(1)[0]
            
            #print(q_values.size(),a.unsqueeze(1).size())
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.gamma * next_q_value * (1 - done)
            # Notice that detach the expected_q_value
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            #self.optimizer.zero_grad()
            loss.backward()
            #print("steps:",i)
            sg_sign = torch.sign(fs0.grad.to(fs0.device)) 

            fs0 = fs0.detach() + alpha * s0 * sg_sign

        fs0.detach_()

        q_values = self.model(fs0)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def pgd(self, epsilon=0.2, steps=2, alpha=0.1):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample()

        fs0 = torch.distributions.uniform.Uniform(s0 - epsilon * s0, s0 + epsilon * s0 + 1e-8).sample()

        for i in range(steps):
            fs0.requires_grad = True
            q_values = self.model(fs0)
            next_q_values = self.model(s1)
            next_q_value = next_q_values.max(1)[0]
            
            #print(q_values.size(),a.unsqueeze(1).size())
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.gamma * next_q_value * (1 - done)
            # Notice that detach the expected_q_value
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            #self.optimizer.zero_grad()
            loss.backward()
            #print("steps:",i)
            sg_sign = torch.sign(fs0.grad.to(fs0.device)) 

            fs0 = fs0.detach() + alpha * s0 * sg_sign

        fs0.detach_()

        q_values = self.model(fs0)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def mim(self, epsilon=0.2, steps=2, alpha=0.1, decay=1.0):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample()

        fs0 = s0.clone()

        fs0_momentum = torch.zeros_like(s0).detach().to(device)

        for i in range(steps):
            fs0.requires_grad = True
            q_values = self.model(fs0)
            next_q_values = self.model(s1)
            next_q_value = next_q_values.max(1)[0]
            
            #print(q_values.size(),a.unsqueeze(1).size())
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.gamma * next_q_value * (1 - done)
            # Notice that detach the expected_q_value
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            #self.optimizer.zero_grad()
            loss.backward()
            #print("steps:",i)
            fs0_grads = fs0.grad.to(fs0.device)
            fs0_grads = fs0_momentum*decay + fs0_grads/torch.norm(fs0_grads,p=1)
            fs0_momentum = fs0_grads
            sg_sign = torch.sign(fs0_grads) 

            fs0 = fs0.detach() + alpha * s0 * sg_sign

        fs0.detach_()

        q_values = self.model(fs0)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def save(self, checkpoint_path):
        checkpoint = { 
        'model': self.nn.state_dict(),
        'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, checkpoint_path)
   
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.nn.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
