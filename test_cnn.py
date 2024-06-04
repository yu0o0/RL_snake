import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv


def main():
    RENDER_DELAY = 0.5
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(silent_mode=False)
    state = env.reset()
   
    sum_reward = 0

    policy=SnakeCNN(12, 3).to(device)
    policy.load_state_dict(torch.load(r"result\ex3\weight\best.pt"))
    
    while True:
        # print(obs.shape)
        # print(policy(obs).shape)
        # plt.imshow(obs, interpolation='nearest')
        # plt.show()
        # print(policy.choose_action(obs))
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = torch.argmax(policy(state_tensor)[0])
        # print(policy(state_tensor))
        # print(action)
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
