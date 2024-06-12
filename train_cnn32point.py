import os
import sys
import random

import numpy as np

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv
from collections import deque

class EpsilonGreedyPolicy:
    '''
    Helper class to create/manage/use epsilon-greedy policies with q
    '''

    def __init__(self, epsilon, epsilon_decay_len, actions, seed=0):
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.epsilon_decay_len = epsilon_decay_len
        # assume number of actions same for all states
        self.actions = list(actions)
        self.num_actions = len(actions)
        self.prng = random.Random()
        self.prng.seed(seed)

        # pre-compute a few things for efficiency
        self.greedy_prob = 1.0-epsilon+epsilon/self.num_actions
        self.rand_prob = epsilon/self.num_actions

    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon0 * \
            (self.epsilon_decay_len - episode)/self.epsilon_decay_len
        if self.epsilon < 0:
            self.epsilon = 0
        self.greedy_prob = 1.0-self.epsilon+self.epsilon/self.num_actions
        self.rand_prob = self.epsilon/self.num_actions

    def choose_action(self, greedy_action):
        '''
        Given q_s=q(state), make epsilon-greedy action choice
        '''
        # create epsilon-greedy policy (at current state only) from q_s
        policy = [self.rand_prob]*self.num_actions
        policy[greedy_action] = self.greedy_prob

        # choose random action based on e-greedy policy
        action = self.prng.choices(self.actions, weights=policy)[0]

        return action



def main():
    DEBUG = False
    
    EPOCHNUM = 1
    NUM_EPISODES = 30000
    showPlots = True
    from matplotlib import pyplot as plt
    
    folder_name = "3A_CNN32p"
    train_idx = 4
    resume = True
    resume_idx = 3
    numActions = 3
    obsSize = 12
    model_CFG = dict(
        numActions=numActions, 
        locPoint=11, 
        imgPoint=32,
        obsSize=obsSize,
        numFeaMap=10
    )
    lr = 0.003
    gamma = 0.9
    epsilon = 0.2
    replay_size = 5000
    batch_size = 64
    
    output_path = f"./result/{folder_name}_{train_idx}"
    resume_path = f"./result/{folder_name}_{resume_idx}/best.pt"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/weight', exist_ok=True)
    
    env = SnakeEnv(seed=100, silent_mode=True, random_episode=False)

    # Double DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = SnakeCNN(**model_CFG).to(device)
    if resume:
        policy_net.load_state_dict(torch.load(resume_path))
    target_net = SnakeCNN(**model_CFG).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_return = -1e10
    episodeLengths = []
    episodeRewards = []
    averagedRewards = []

    for epoch in range(EPOCHNUM):
        # ε-貪婪策略
        epsilon_decay_len = NUM_EPISODES
        actions = list(range(numActions))
        egp = EpsilonGreedyPolicy(epsilon, epsilon_decay_len, actions, seed=55)

        replayBuffer=deque(maxlen=replay_size)

        for episode in range(NUM_EPISODES):

            loc, img = env.reset()

            egp.decay_epsilon(episode)
            
            while True:
                policy_net.train()
                # 1. From current state , take action according to -greedy policy
                with torch.no_grad():
                    loc_tensor = torch.tensor(loc, dtype=torch.float).to(device).unsqueeze(0)
                    img_tensor = torch.tensor(img, dtype=torch.float).to(device).unsqueeze(0)
                    action_probs = policy_net.choose_action(loc_tensor, img_tensor)
                    # action = torch.argmax(q_s[0])
                    action = egp.choose_action(action_probs[0])
                    (next_loc, next_img), reward, done, info = env.step(action)
                
                # 2. Store experience in replay memory 經驗回放
                replayBuffer.append((loc, img, action, reward, next_loc, next_img, done))

                # 3. Sample random mini-batch of experiences from replay memory
                minibatch = random.sample(replayBuffer, batch_size) if len(replayBuffer) > batch_size else replayBuffer

                # 4. Update weights using semi-gradient Q-learning update rule
                # Creating a tensor from a list of numpy.ndarrays is extremely slow.
                locs, imgs, actions, rewards, next_locs, next_imgs, dones = zip(*minibatch)
                locs = torch.tensor(np.array(locs), dtype=torch.float).to(device)
                imgs = torch.tensor(np.array(imgs), dtype=torch.float).to(device)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards).to(device)
                next_locs = torch.tensor(np.array(next_locs), dtype=torch.float32).to(device)
                next_imgs = torch.tensor(np.array(next_imgs), dtype=torch.float32).to(device)
                dones = torch.tensor(dones).to(device)
                

                state_action_values = policy_net(locs, imgs).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_state_values = target_net(next_locs, next_imgs).max(1)[0]
                    expected_state_action_values = rewards + gamma * next_state_values * (~dones)
                
                if DEBUG:
                    print(actions)
                    print(state_action_values)
                    print(expected_state_action_values)
                    plt.imshow(imgs[-1].to("cpu"), interpolation='nearest')
                    plt.show()
                
                
                optimizer.zero_grad()
                loss = criterion(state_action_values, expected_state_action_values)
                loss.backward()
                optimizer.step()

                if done: 
                    break 
                loc, img = next_loc, next_img
            
            # 驗證
            policy_net.eval()
            tot_reward = 0
            loc, img = env.reset()
            step = 0
            while True:
                with torch.no_grad():
                    loc_tensor = torch.tensor(loc, dtype=torch.float).to(device).unsqueeze(0)
                    img_tensor = torch.tensor(img, dtype=torch.float).to(device).unsqueeze(0)
                    q_s = policy_net(loc_tensor, img_tensor)
                    action = torch.argmax(q_s[0])
                    (loc, img), reward, done, info = env.step(action)
                    tot_reward += reward
                    step+=1
                    if done: break 
            if tot_reward > best_return:
                best_return = tot_reward
                print(
                    f"New best weights found @ epoch: {epoch+1} , episode:{episode+1} tot_reward:{tot_reward}")
                print(f"step: {step}")
                torch.save(policy_net.state_dict(), f'{output_path}/best.pt')
                
            if episode % 100 == 0:
                torch.save(policy_net.state_dict(), f'{output_path}/last.pt')
                
            target_net.load_state_dict(policy_net.state_dict())
            
            # update stats for later plotting
            window_len = 100
            episodeLengths.append(step)
            episodeRewards.append(tot_reward)
            w = len(episodeRewards) if len(episodeRewards)<window_len else window_len
            avg_tot_reward = sum(episodeRewards[-w:])/w
            averagedRewards.append(avg_tot_reward)

            # if episode % 100 == 0:
            print(f'epoch: {epoch+1} , episode: {episode+1}\ttotal reward: {tot_reward}')

    if showPlots:
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.subplot(312)
        plt.plot(episodeRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.subplot(313)
        plt.plot(averagedRewards)
        plt.xlabel('Episode')
        plt.ylabel('Avg Total Reward')
        plt.savefig(f"{output_path}/training INFO.png")
        plt.show()
        # cleanup plots
        plt.cla()
        plt.close('all')
        
    env.close()


if __name__ == "__main__":
    main()
