import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
#print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    #print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    pass
    #print("Device set to : cpu")
#print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def argsort(self,seq):
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
        
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, nagents, state_dim, action_dim, has_continuous_action_space, action_std_init, device=device):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        size = 256
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, size),
                            nn.Tanh(),
                            nn.Linear(size, 2*size),
                            nn.Tanh(),
                            nn.Linear(2*size, size),
                            nn.Tanh(),
                            nn.Linear(size, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, size),
                            nn.Tanh(),
                            nn.Linear(size, 2*size),
                            nn.Tanh(),
                            nn.Linear(2*size, size),
                            nn.Tanh(),
                            nn.Linear(size, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim*nagents, size),
                        nn.Tanh(),
                        nn.Linear(size, 2*size),
                        nn.Tanh(),
                        nn.Linear(2*size, size),
                        nn.Tanh(),
                        nn.Linear(size, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def act2(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        #action = dist.sample()
        action = torch.argmax(dist.probs)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def act3(self, state, action0):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        #action = dist.sample()
        action = torch.tensor(action0).to(device)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, observations, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(observations)
        
        return action_logprobs, state_values, dist_entropy
    
    def converge_eval(self, state, threshold=0.6):
        idx = 0
        converge_idx = -1
        while state[0][-1]==0:
            action_probs = self.actor(state)
            temp = torch.argmax(action_probs)
            if action_probs[0][temp] > threshold:
                converge_idx = idx
            state[0][idx] = temp+1
            idx += 1
        return converge_idx
    
class PPO:
    def __init__(self, nagents, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6,device=device):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.pbuffer = dict() #permanant buffer
        

        self.policy = ActorCritic(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init,device=device).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ], amsgrad=True)#, weight_decay=1e-4, betas=(0.85, 0.99)

        self.policy_old = ActorCritic(nagents, state_dim, action_dim, has_continuous_action_space, action_std_init,device=device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, obs, idx):
        state = obs[idx]      
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act(state)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act(state)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def select_action3(self, obs, idx, action0):
        state = obs[idx]      
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act3(state, action0)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act3(state, action0)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def select_action2(self, obs, idx):
        state = obs[idx]      
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act2(state)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                obs = torch.FloatTensor(obs.flatten()).to(device)
                action, action_logprob = self.policy_old.act2(state)
            self.buffer.observations.append(obs)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def fgs(self, epsilon=0.2):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        for i in range(len(self.buffer.states)):
            try:
                rewards[i] = self.pbuffer[self.buffer.states[i]] = max(self.pbuffer[self.buffer.states[i]],rewards[i])#rewards[i] = 
            except KeyError:
                self.pbuffer[self.buffer.states[i]] = rewards[i]
                
        #rewards_all = torch.tensor(list(self.pbuffer.values()), dtype=torch.float32).to(device)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) #use global mean and std

        # convert list to tensor
        old_observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).clone().requires_grad_().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).clone().requires_grad_().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        observations_grads = []
        states_grads = []
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_observations, old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            #self.optimizer.zero_grad()
                
            loss.mean().backward()           

            observations_grads.append(old_observations.grad.to(old_observations.device))
            states_grads.append(old_states.grad.to(old_states.device))
            
            
        og_sign = torch.sign(torch.sum(torch.stack(observations_grads), dim=0))
        sg_sign = torch.sign(torch.sum(torch.stack(states_grads), dim=0))
                
        fake_observations = old_observations + epsilon * old_observations * og_sign

        fake_states = old_states + epsilon * old_states * sg_sign
            
        fake_observations.detach_()

        fake_states.detach_()

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
                
        #return fake_observations, fake_states

    def bim(self, epsilon=0.2, steps=2, alpha=0.1):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        for i in range(len(self.buffer.states)):
            try:
                rewards[i] = self.pbuffer[self.buffer.states[i]] = max(self.pbuffer[self.buffer.states[i]],rewards[i])#rewards[i] = 
            except KeyError:
                self.pbuffer[self.buffer.states[i]] = rewards[i]
                
        #rewards_all = torch.tensor(list(self.pbuffer.values()), dtype=torch.float32).to(device)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) #use global mean and std

        # convert list to tensor
        fake_observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).clone().to(device)
        fake_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).clone().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        old_observations = fake_observations.clone().detach().to(device)
        old_states = fake_states.clone().detach().to(device)

        for i in range(steps):
            fake_observations.requires_grad = True
            fake_states.requires_grad = True
            # Optimize policy for K epochs
            observations_grads = []
            states_grads = []
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                
                # take gradient step
                #self.optimizer.zero_grad()
                    
                loss.mean().backward()           

                observations_grads.append(fake_observations.grad.to(fake_observations.device))
                states_grads.append(fake_states.grad.to(fake_states.device))
                
                
            og_sign = torch.sign(torch.sum(torch.stack(observations_grads), dim=0))
            sg_sign = torch.sign(torch.sum(torch.stack(states_grads), dim=0))
                    
            fake_observations = fake_observations.detach() + alpha * old_observations * og_sign
            fake_states = fake_states.detach() + alpha * old_states * sg_sign
            
        fake_observations.detach_()

        fake_states.detach_()

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
                
        #return fake_observations, fake_states

    def pgd(self, epsilon=0.2, steps=2, alpha=0.1):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        for i in range(len(self.buffer.states)):
            try:
                rewards[i] = self.pbuffer[self.buffer.states[i]] = max(self.pbuffer[self.buffer.states[i]],rewards[i])#rewards[i] = 
            except KeyError:
                self.pbuffer[self.buffer.states[i]] = rewards[i]
                
        #rewards_all = torch.tensor(list(self.pbuffer.values()), dtype=torch.float32).to(device)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) #use global mean and std

        # convert list to tensor
        old_observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).clone().detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).clone().detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        fake_observations = torch.distributions.uniform.Uniform(old_observations - epsilon * old_observations, old_observations + epsilon * old_observations + 1e-8).sample().to(device)
        fake_states = torch.distributions.uniform.Uniform(old_states - epsilon * old_states, old_states + epsilon * old_states + 1e-8).sample().to(device)

        for i in range(steps):
            fake_observations.requires_grad = True
            fake_states.requires_grad = True
            # Optimize policy for K epochs
            observations_grads = []
            states_grads = []
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                
                # take gradient step
                #self.optimizer.zero_grad()
                    
                loss.mean().backward()           

                observations_grads.append(fake_observations.grad.to(fake_observations.device))
                states_grads.append(fake_states.grad.to(fake_states.device))
                
                
            og_sign = torch.sign(torch.sum(torch.stack(observations_grads), dim=0))
            sg_sign = torch.sign(torch.sum(torch.stack(states_grads), dim=0))
                    
            fake_observations = fake_observations.detach() + alpha * old_observations * og_sign

            fake_states = fake_states.detach() + alpha * old_states * sg_sign
            
        fake_observations.detach_()

        fake_states.detach_()

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
                
        #return fake_observations, fake_states

    def mim(self, epsilon=0.2, steps=2, alpha=0.1, decay=1.0):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        for i in range(len(self.buffer.states)):
            try:
                rewards[i] = self.pbuffer[self.buffer.states[i]] = max(self.pbuffer[self.buffer.states[i]],rewards[i])#rewards[i] = 
            except KeyError:
                self.pbuffer[self.buffer.states[i]] = rewards[i]
                
        #rewards_all = torch.tensor(list(self.pbuffer.values()), dtype=torch.float32).to(device)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) #use global mean and std

        # convert list to tensor
        fake_observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).clone().to(device)
        fake_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).clone().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        old_observations = fake_observations.clone().detach().to(device)
        old_states = fake_states.clone().detach().to(device)

        observations_momentum = torch.zeros_like(old_observations).detach().to(device)
        states_momentum = torch.zeros_like(old_states).detach().to(device)

        for i in range(steps):
            fake_observations.requires_grad = True
            fake_states.requires_grad = True
            # Optimize policy for K epochs
            observations_grads = []
            states_grads = []
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
                
                # take gradient step
                #self.optimizer.zero_grad()
                    
                loss.mean().backward()           

                observations_grads.append(fake_observations.grad.to(fake_observations.device))
                states_grads.append(fake_states.grad.to(fake_states.device))
                
            observations_grads = torch.sum(torch.stack(observations_grads), dim=0)
            observations_grads = observations_momentum*decay + observations_grads/torch.norm(observations_grads,p=1)
            observations_momentum = observations_grads

            states_grads = torch.sum(torch.stack(states_grads), dim=0)
            states_grads = states_momentum*decay + states_grads/torch.norm(states_grads,p=1)
            states_momentum = states_grads


            og_sign = torch.sign(observations_grads)
            sg_sign = torch.sign(states_grads)
                    
            fake_observations = fake_observations.detach() + alpha * old_observations * og_sign
            fake_states = fake_states.detach() + alpha * old_states * sg_sign
            
        fake_observations.detach_()

        fake_states.detach_()

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(fake_observations, fake_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
                
        #return fake_observations, fake_states

    def update(self, threshold=0.6):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        for i in range(len(self.buffer.states)):
            try:
                rewards[i] = self.pbuffer[self.buffer.states[i]] = max(self.pbuffer[self.buffer.states[i]],rewards[i])#rewards[i] = 
            except KeyError:
                self.pbuffer[self.buffer.states[i]] = rewards[i]
                
        #rewards_all = torch.tensor(list(self.pbuffer.values()), dtype=torch.float32).to(device)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) #use global mean and std

        # convert list to tensor
        old_observations = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_observations, old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        #init_state = torch.FloatTensor([[0]*8]).detach().to(device)
        
        #convergence_eval = self.policy.converge_eval(init_state, threshold=threshold)
        
        #return convergence_eval        
    
    def save(self, checkpoint_path):
        checkpoint = { 
        'model': self.policy_old.state_dict(),
        'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, checkpoint_path)
   
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(checkpoint['model'])
        self.policy.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
