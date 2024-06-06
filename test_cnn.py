import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv


def main():
    DEBUG = False
    RENDER_DELAY = 0.2
    numActions = 4
    obsSize = 12
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(silent_mode=False, seed=0)
    state = env.reset()

    sum_reward = 0

    policy=SnakeCNN(obsSize, numActions).to(device)
    # policy.load_state_dict(torch.load(r"result\ex6\weight\last.pt"))
    policy.load_state_dict(torch.load(r"result\ex6\weight\best.pt"))
    policy.eval()
    while True:
        # print(obs.shape)
        # print(policy(obs).shape)
        
        # print(policy.choose_action(obs))
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        q_s = policy(state_tensor)
        action = torch.argmax(q_s[0])
        
        if DEBUG:
            print(q_s)
            print(action)
            plt.imshow(state, interpolation='nearest')
            plt.show()
        
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if np.absolute(reward) > 0.001:
            print(reward)
        env.render()
        time.sleep(RENDER_DELAY)
        
        if done: break 
        
    # print(info["snake_length"])
    # print(info["food_pos"])
    # print(obs)
    print("sum_reward: %f" % sum_reward)
    print("episode done")
    time.sleep(1)
    
    env.close()
    print(f"Total reward: {sum_reward}")



if __name__ == "__main__":
    main()
