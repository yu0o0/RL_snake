import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv

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

    def choose_action(self, q_s):
        '''
        Given q_s=q(state), make epsilon-greedy action choice
        '''
        # create epsilon-greedy policy (at current state only) from q_s
        policy = [self.rand_prob]*self.num_actions
        with torch.no_grad():
            greedy_action = torch.argmax(q_s)
        policy[greedy_action] = self.greedy_prob

        # choose random action based on e-greedy policy
        action = self.prng.choices(self.actions, weights=policy)[0]

        return action



def main():
    NUM_EPISODES = 30000
    max_episode_len = 100
    showPlots = True
    from matplotlib import pyplot as plt
    
    numActions = 3
    obsSize = 12
    lr = 0.001
    gamma = 0.99
    folder_name = "ex2"
    output_path = f"./result/{folder_name}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/weight', exist_ok=True)
    
    env = SnakeEnv(silent_mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = SnakeCNN(state_size, action_size).to(device)
    target_net = SnakeCNN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    

    best_return = -1e10
    episodeLengths = []
    episodeRewards = []
    averagedRewards = []

    epsilon_decay_len = NUM_EPISODES
    actions = list(range(numActions))
    egp = EpsilonGreedyPolicy(1.0, epsilon_decay_len, actions, seed=0)

    for episode in range(NUM_EPISODES):
        if episode % 100 == 0:
            print('Episode: {}'.format(episode+1))
            
        state = env.reset()
        sum_reward = 0
        bootstrap_record = []
        optimizer.zero_grad()

        egp.decay_epsilon(episode)
        
        step = 0
        while True:
            
            (action, log_pi) = policy.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            bootstrap_record.append((log_pi, reward))
            sum_reward += reward
            step+=1

            if done: break 
            state = next_state
            
        Gt = 0
        for t, (ln_pi, reward) in list(enumerate(bootstrap_record))[::-1]:
            Gt = reward+gamma*Gt

            # γGt ∇ln π(St, At, θ) = ∇(γGt * ln π(St, At, θ))
            # Hope it's larger the better, opposite to loss
            (-gamma**(t)*Gt*ln_pi).backward()
            
        optimizer.step()

        if sum_reward > best_return:
            best_return = sum_reward
            print(
                f"New best weights found @ episode:{episode+1} tot_reward:{sum_reward}")
            torch.save(policy.state_dict(),
                       f'{output_path}/weight/best.pt')

        # update stats for later plotting
        window_len = 100
        episodeLengths.append(step)
        episodeRewards.append(sum_reward)
        w = len(episodeRewards) if len(episodeRewards)<window_len else window_len
        avg_tot_reward = sum(episodeRewards[-w:])/w
        averagedRewards.append(avg_tot_reward)

        if episode % 100 == 0:
            print('\tAvg reward: {}'.format(avg_tot_reward))

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
