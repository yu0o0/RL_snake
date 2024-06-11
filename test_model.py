import os
import sys
import random

import torch
from snake_env import SnakeEnv
from snake_model import SnakeCNN, SnakeFCN

def main():
    DEBUG = False
    RENDER_DELAY = 0.2
    numActions = 3
    obsSize = 12
    folder_name = "3A_CNN9p"
    # folder_name = "3A_FCN"
    weight_idx = 1
    weight_type = (
        "best"
        # "last"
        )
    model_CFG = dict(
        numActions=numActions, 
        locPoint=11, 
        imgPoint=9,
        obsSize=obsSize,
        numFeaMap=10
    )
    policy_net=SnakeCNN(**model_CFG)
    # policy_net=SnakeFCN(**model_CFG)
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    weight_path = f"./result/{folder_name}_{weight_idx}/{weight_type}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net=policy_net.to(device)
    policy_net.load_state_dict(torch.load(weight_path))
    policy_net.eval()

    env = SnakeEnv(silent_mode=False, seed=55)
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
