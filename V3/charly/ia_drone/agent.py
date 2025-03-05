import torch
import random
import numpy as np
from collections import deque
from game import AckermannAgent, Environment
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(5, 256, 3)  # 5 inputs (x, y, theta, goal_x - x, goal_y - y), 3 outputs (actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, agent, goal_x, goal_y):
        state = agent.get_state(goal_x, goal_y)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = Environment()
    while True:
        # get old state
        state_old = agent.get_state(env.agent, env.goal_x, env.goal_y)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        delta = final_move.index(1) - 1  # Convert action to delta
        env.agent.update_position(delta)
        reward = env.get_reward()
        state_new = agent.get_state(env.agent, env.goal_x, env.goal_y)

        # check if done
        done = env.is_collision(env.agent.x, env.agent.y, 10) or env.agent.distance_to_goal(env.goal_x, env.goal_y) < 15

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print('Game', agent.n_games, 'Score', reward, 'Record:', record)

            plot_scores.append(reward)
            total_score += reward
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        # Update the environment display
        env.update()

if __name__ == '__main__':
    train()
