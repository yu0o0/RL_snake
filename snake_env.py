import math

import gym
import numpy as np

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(
            seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(3)  # 0: 向左, 1: 向前, 2: 向右

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2  # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            # More than enough steps to get the food.
            self.step_limit = int(self.grid_size * 0.5)
        else:
            self.step_limit = 1e9  # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0
        self.hungry = False

        state = self._generate_observation()
        return state

    def step(self, action):
        # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        self.done, info = self.game.step(action)
        state = self._generate_observation()

        reward = 0
        self.reward_step_counter += 1

        # Snake fills up the entire board. Game over.
        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1  # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return state, reward, self.done, info

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.hungry = True
            self.done = True

        if self.done:  # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)
            reward -= 20
            if self.hungry:
                reward *= 0
            else:
                reward *= 3

            return state, reward, self.done, info

        elif info["food_obtained"]:  # Food eaten. Reward boost on snake size.
            # reward = info["snake_size"] * 100 / self.grid_size
            reward += 30
            self.reward_step_counter = 0  # Reset reward step counter

        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                # reward += 1 / info["snake_size"]
                reward += 0.5
            else:
                reward -= 0.5
                # reward -= 2 / info["snake_size"]
            reward *= 2
            # reward -= self.reward_step_counter * 0.005

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return state, reward, self.done, info

    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return False
            else:
                col += 1

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                # The snake won't pop the last cell if it ate food.
                (row, col) in snake_list
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                # The snake will pop the last cell if it did not eat food.
                (row, col) in snake_list[:-1]
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        img = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        img[tuple(np.transpose(self.game.snake))] = np.linspace(
            255, 50, len(self.game.snake), dtype=np.uint8)
        img = np.expand_dims(img, axis=2)

        loc = [
            # Food location
            self.game.food[0] - self.game.snake[0][0],  # food left
            # self.game.food[0] > self.game.snake[0][0],  # food right
            self.game.food[1] - self.game.snake[0][1],  # food up
            # self.game.food[1] > self.game.snake[0][1]  # food down

            self.game.direction == "LEFT",  #蛇的面向(左)
            self.game.direction == "RIGHT", #蛇的面向(右)
            self.game.direction == "UP",    #蛇的面向(上)
            self.game.direction == "DOWN",  #蛇的面向(下)

            self.reward_step_counter,
        ]
        loc = np.array(loc)

        return loc, img


if __name__ == "__main__":
    # Test the environment using random actions
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
        # print(info["snake_length"])
        # print(info["food_pos"])
        # print(obs)
        print("sum_reward: %f" % sum_reward)
        print("episode done")
        # time.sleep(100)

    env.close()
    print("Average episode reward for random strategy: {}".format(
        sum_reward/NUM_EPISODES))
