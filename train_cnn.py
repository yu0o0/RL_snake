import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv


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
    
    env = SnakeEnv(silent_mode=True)
    policy=SnakeCNN(obsSize, numActions)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=lr)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/weight', exist_ok=True)
    
    best_return = -1e10
    episodeLengths = []
    episodeRewards = []
    averagedRewards = []

    for episode in range(NUM_EPISODES):
        if episode % 100 == 0:
            print('Episode: {}'.format(episode+1))
            
        state = env.reset()
        sum_reward = 0
        bootstrap_record = []
        optimizer.zero_grad()
        
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
