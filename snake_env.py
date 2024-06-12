import math

import gym
import numpy as np

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True, random_episode=False):
        super().__init__()
        self.game = SnakeGame(
            seed=seed, board_size=board_size, silent_mode=silent_mode, random_episode=random_episode)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(3)  # 0: 向左, 1: 向前, 2: 向右

        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = int(self.grid_size)
        else:
            self.step_limit = 1e9
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0
        self.hungry = False

        state = self._generate_observation()
        return state

    def step(self, action):
        self.done, info = self.game.step(action)
        state = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1
            self.done = True
            return state, reward, self.done, info

        if self.reward_step_counter > self.step_limit: 
            self.reward_step_counter = 0
            self.hungry = True
            self.done = True

        if self.done: 
            reward -= 20
            if self.hungry:
                reward *= 2
            else:
                reward *= 1

            return state, reward, self.done, info

        elif info["food_obtained"]: 
            reward += 20
            self.reward_step_counter = 0

        else:
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward += 1
            else:
                reward -= 1
            reward *= 0.001

        return state, reward, self.done, info

    def render(self):
        self.game.render()

    def _generate_observation(self):
        img = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        img[tuple(np.transpose(self.game.snake))] = 125
        img[tuple(self.game.snake[0])] = 255
        img = np.expand_dims(img, axis=2)

        head = self.game.snake[0]
        row, col = head
        life = self.step_limit - self.reward_step_counter
        dir_l = self.game.direction == "LEFT"  # 蛇向(左)
        dir_r = self.game.direction == "RIGHT" # 蛇向(右)
        dir_u = self.game.direction == "UP"    # 蛇向(上)
        dir_d = self.game.direction == "DOWN"  # 蛇向(下)
        point_l = (row, col-1)  #往左的點座標
        point_r = (row, col+1)  #往右的點座標
        point_u = (row-1, col)  #往上的點座標
        point_d = (row+1, col)  #往下的點座標
        loc = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or 
            (dir_l and self.game.is_collision(point_l)) or 
            (dir_u and self.game.is_collision(point_u)) or 
            (dir_d and self.game.is_collision(point_d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or 
            (dir_d and self.game.is_collision(point_l)) or 
            (dir_l and self.game.is_collision(point_u)) or 
            (dir_r and self.game.is_collision(point_d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or 
            (dir_u and self.game.is_collision(point_l)) or 
            (dir_r and self.game.is_collision(point_u)) or 
            (dir_l and self.game.is_collision(point_d)),
            
            # Food location
            self.game.food[0] - self.game.snake[0][0],
            self.game.food[1] - self.game.snake[0][1],

            self.game.direction == "LEFT",  #蛇的面向(左)
            self.game.direction == "RIGHT", #蛇的面向(右)
            self.game.direction == "UP",    #蛇的面向(上)
            self.game.direction == "DOWN",  #蛇的面向(下)

            life,
            len(self.game.snake),
        ]
        loc = np.array(loc)

        return loc, img


if __name__ == "__main__":
    NUM_EPISODES = 100
    RENDER_DELAY = 0.001
    from matplotlib import pyplot as plt
    import time

    env = SnakeEnv(silent_mode=False)

    sum_reward = 0

    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(state)
            plt.imshow(state[1])
            plt.show()
            sum_reward += reward
            if np.absolute(reward) > 0.001:
                print(reward)
            env.render()

            time.sleep(RENDER_DELAY)
        print("sum_reward: %f" % sum_reward)
        print("episode done")

    env.close()
    print("Average episode reward for random strategy: {}".format(
        sum_reward/NUM_EPISODES))
