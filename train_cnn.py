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
    folder_name = "ex1"
    output_path = f"./result/{folder_name}"
    
    env = SnakeEnv(silent_mode=False)
    policy=SnakeCNN(obsSize, numActions)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=lr)
    os.makedirs(output_path, exist_ok=True)
    
    sum_reward = 0

    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        while not done:
            plt.imshow(obs, interpolation='nearest')
            plt.show()
            action = env.action_space.sample()
            # action = action_list[i]
            i = (i + 1) % len(action_list)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if np.absolute(reward) > 0.001:
                print(reward)
            env.render()
            
            time.sleep(RENDER_DELAY)
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
