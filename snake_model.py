import collections
import torch


class SnakeCNN(torch.nn.Module):
    def __init__(self, obsSize, numActions):
        '''
        Parameters:
            numFeatures: Number of input features
            numActions: Number of output actions
        '''
        super().__init__()

        self.IMG_C1 = torch.nn.Sequential(collections.OrderedDict([
            ('c', torch.nn.Conv2d(1, 10, kernel_size=(3, 3), padding=(1, 1))),
            # ('MaxPool', torch.nn.MaxPool2d(kernel_size=7, stride=2))
            # ('ReLU', torch.nn.ReLU()),
        ]))
        self.IMG_F1 = torch.nn.Sequential(collections.OrderedDict([
            ('f', torch.nn.Linear(1440, 9)),
            ('ReLU', torch.nn.ReLU()),
            # ('dropout', torch.nn.Dropout(p=0.5))
        ]))
        self.LOC_F1 = torch.nn.Sequential(collections.OrderedDict([
            ('f', torch.nn.Linear(20, 256)),
            ('ReLU', torch.nn.ReLU()),
            # ('dropout', torch.nn.Dropout(p=0.5))
        ]))
        self.LOC_F2 = torch.nn.Sequential(collections.OrderedDict([
            ('f', torch.nn.Linear(256, numActions)),
            # ('ReLU', torch.nn.ReLU()),
            # ('dropout', torch.nn.Dropout(p=0.5))
        ]))

        self.MIX1 = torch.nn.Sequential(collections.OrderedDict([
            ('f', torch.nn.Linear(30, 256)),
            ('ReLU', torch.nn.ReLU()),
            # ('dropout', torch.nn.Dropout(p=0.5))
        ]))
        self.MIX_OUT = torch.nn.Sequential(collections.OrderedDict([
            ('f', torch.nn.Linear(256, numActions)),
        ]))

        self.softmax=torch.nn.Softmax(dim=1)
        

    def forward(self, locs, imgs):
        '''
        loc: B, 11
        img: B, 12, 12, 1
        '''
        imgs = imgs.permute(0, 3, 1, 2) # B, 1, 12, 12
        imgs_t = self.IMG_C1(imgs)  # B, 64, 1, 1
        imgs_t = torch.flatten(imgs_t, 1)   # B, 64
        imgs_t = self.IMG_F1(imgs_t)    # B, 5
        mix = torch.cat((locs, imgs_t), dim=1)  # B, 16
        mix = self.LOC_F1(mix)  # B, 256
        mix = self.LOC_F2(mix)  # B, numActions
        
        
        
        
        # locs_t = self.LOC_F1(locs)  # B, 24

        # channel 維度移到高寬前面
        # imgs = imgs.permute(0, 3, 1, 2)
        # imgs_t = self.IMG_C1(imgs)  # B, 10, 5, 5
        # imgs_t = torch.flatten(imgs_t, 1)   # B, 250
        # imgs_t = self.IMG_F1(imgs_t)    # B, 10

        # mix = torch.cat((locs_t, imgs_t), dim=1)  # B, 30

        # mix = self.MIX1(mix)    # B, 256
        # mix = self.MIX_OUT(mix) # B, numActions

        return mix

    def choose_action(self, locs, imgs, return_q_s=False):
        '''
        Sample an action at state s, using current policy

        Returns chosen action, and optionally the computed PG log pi term
        '''
        q_s = self.forward(locs, imgs)
        probs = self.softmax(q_s)
        actions = torch.multinomial(probs, 1)
        
        if return_q_s:
            return actions, q_s
        else:
            return actions
        


if __name__ == "__main__":
    # Test the environment using random actions
    NUM_EPISODES = 100
    RENDER_DELAY = 0.001
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(silent_mode=False)

    sum_reward = 0

    policy = SnakeCNN(12, 3).to(device)

    for _ in range(NUM_EPISODES):
        loc, img = env.reset()
        done = False
        while not done:
            loc_tensor = torch.tensor(loc, dtype=torch.float).to(device).unsqueeze(0)
            img_tensor = torch.tensor(img, dtype=torch.float).to(device).unsqueeze(0)
            
            # print(policy.choose_action(obs))
            actions, q_s = policy.choose_action(loc_tensor, img_tensor, return_q_s=True)
            action = actions[0]
            (loc, img), reward, done, info = env.step(action)
            sum_reward += reward
            if np.absolute(reward) > 0.001:
                print(reward)
            env.render()

            print(q_s, action)
            plt.imshow(img)
            plt.show()

            time.sleep(RENDER_DELAY)
        # print(info["snake_length"])
        # print(info["food_pos"])
        # print(obs)
        print("sum_reward: %f" % sum_reward)
        print("episode done")
        # time.sleep(100)

    env.close()
    print("Average episode reward for random strategy: {}".format(
        sum_reward/NUM_EPISODES))
