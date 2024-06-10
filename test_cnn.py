import os
import sys
import random

import torch
from snake_model import SnakeCNN
from snake_env import SnakeEnv


def main():
    DEBUG = False
    RENDER_DELAY = 0.2
    numActions = 3
    obsSize = 12
    folder_name = "MIX-3A_3filter9p_2"
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # policy_net=SnakeCNN(obsSize, numActions).to(device)
    # policy_net.load_state_dict(torch.load(f"result/{folder_name}/weight/last.pt"))
    # policy_net.load_state_dict(torch.load(f"result/{folder_name}/weight/best.pt"))
    policy_net = torch.load(f"result/{folder_name}/weight/best_model.pt").to(device)
    policy_net.eval()

    env = SnakeEnv(silent_mode=False, seed=0)
    loc, img = env.reset()

    sum_reward = 0
    while True:
        loc_tensor = torch.tensor(loc, dtype=torch.float).to(device).unsqueeze(0)
        img_tensor = torch.tensor(img, dtype=torch.float).to(device).unsqueeze(0)
        q_s = policy_net(loc_tensor, img_tensor)
        action = torch.argmax(q_s[0])
        
        if DEBUG:
            print(q_s)
            print(action)
            plt.imshow(img)
            plt.show()
        
        (loc, img), reward, done, info = env.step(action)
        sum_reward += reward
        if np.absolute(reward) > 0.001:
            print(reward)
        env.render()
        time.sleep(RENDER_DELAY)
        
        if done: break 
        
    # print(info["snake_length"])
    # print(info["food_pos"])
    # print(obs)
    print("episode done")
    time.sleep(1)
    
    env.close()
    print(f"Total reward: {sum_reward}")



if __name__ == "__main__":
    main()
