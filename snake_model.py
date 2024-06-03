import collections
import torch



class SnakeCNN(torch.nn.Module):
    def __init__(self, obsSize, numActions, useTanh=False):
        '''
        Parameters:
            numFeatures: Number of input features
            numActions: Number of output actions
        '''
        super().__init__()

        self.C1 = torch.nn.Sequential(collections.OrderedDict([
            ('c1', torch.nn.Conv2d(3, 6, kernel_size=(3, 3), padding=(1,1))),
            ('c1_relu', torch.nn.ReLU()),
        ]))
        
        self.dense1=torch.nn.Linear(obsSize**2*6,obsSize**2)
        self.nonlin1=torch.nn.Tanh()
        self.dense2=torch.nn.Linear(obsSize**2,numActions)
        self.softmax1=torch.nn.Softmax(dim=1)
 
    def forward(self,s):
        '''
        Compute policy function pi(a|s,w) by forward computation through MLP   
        '''
        feature_input=torch.tensor(s,dtype=torch.float32)
        feature_input = feature_input.permute(2, 0, 1).unsqueeze(0)

        output = self.C1(feature_input)
        output = torch.flatten(output, 1)
        output=self.dense1(output)
        output=self.nonlin1(output)
        output=self.dense2(output)
        output=self.softmax1(output)

        return output

    def choose_action(self,s,returnLogpi=True):
        '''
        Sample an action at state s, using current policy
        
        Returns chosen action, and optionally the computed PG log pi term
        '''
        pi_s = self.forward(s)
        # print(pi_s)
        prob_model = torch.distributions.Categorical(pi_s)
        action = prob_model.sample()   #sample an action from current policy probabilities
        # print(action)
        
        if not returnLogpi:
            return action.item()
        else:
            log_pi=torch.log(pi_s[0][action]) #log pi
            return (action.item(), log_pi)


if __name__ == "__main__":
    # Test the environment using random actions
    NUM_EPISODES = 100
    RENDER_DELAY = 0.001
    from matplotlib import pyplot as plt
    import time
    import numpy as np
    from snake_env import SnakeEnv

    env = SnakeEnv(silent_mode=False)
    
    num_success = 0
    for i in range(NUM_EPISODES):
        num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    sum_reward = 0

    policy=SnakeCNN(12, 3)
    
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        while not done:
            # print(obs.shape)
            # print(policy(obs).shape)
            plt.imshow(obs, interpolation='nearest')
            plt.show()
            # print(policy.choose_action(obs))
            action, ln_pi = policy.choose_action(obs)
            # action = action_list[i]
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
