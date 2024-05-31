import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv


def main():
    NUM_EPISODES = 100
    RENDER_DELAY = 0.001
    from matplotlib import pyplot as plt
    
    numActions = 3
    obsSize = 12
    lr = 0.001
    gamma = 0.99
    folder_name = "ex1"
    output_path = f"./result/{folder_name}"
    
    env = SnakeEnv(silent_mode=False)
    policy=SnakeCNN(obsSize, numActions)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=lr)
    os.makedirs(output_path, exist_ok=True)
    rewards=[]  
    log_pis=[]
    # sum_reward = 0

    for _ in range(NUM_EPISODES):
        state = env.reset()
        done = False
        # i = 0
        sum_reward = 0
        while not done:
            plt.imshow(state, interpolation='nearest')
            plt.show()
            (action, log_pi) = policy.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            sum_reward += reward
            rewards.append(reward)
            log_pis.append(log_pi)

            if done: break 
            state = next_state
        tot_return = 0
        tot_returns=torch.zeros(len(rewards), dtype=torch.float32) # clean gradient

        for i in reversed(range(len(rewards))): # reward to go
            tot_return = rewards[i] + tot_return*gamma
            tot_returns[i] = tot_return
        # print(tot_returns)

        for t in range(len(tot_returns)): # pseudocode
            tot_returns[t] = (gamma ** t) * tot_returns[t]
        # print(tot_returns)

        logpi_stack=torch.stack(log_pis)
        gradient=torch.dot(tot_returns, logpi_stack) # sum(times) = dot
        gradient.backward()
        # theta = theta + alpha * gradient
        optimizer.step() # Adam not SGD
        optimizer.zero_grad()
        # print(info["snake_length"])
        # print(info["food_pos"])
        # print(obs)
        print("sum_reward: %f" % sum_reward)
        print("episode done")
        # time.sleep(100)
    
    env.close()
    print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))

if __name__ == "__main__":
    main()
