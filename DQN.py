import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 50
MEMORY_CAPACITY = 2000

env = gym.make('CartPole-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        q_action = self.out(x)
        return q_action


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.learn_step_counter = TARGET_REPLACE_ITER

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
        if np.random.uniform() < EPSILON:
            action = torch.max(self.eval_net(s), 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, r, a, s_):
        mem = np.hstack((s, r, a, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = mem
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s = torch.FloatTensor(self.memory[sample_index, :N_STATES])
        b_r = torch.FloatTensor(self.memory[sample_index, N_STATES:N_STATES + 1])
        b_a = torch.LongTensor(self.memory[sample_index, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(self.memory[sample_index, -N_STATES:])

        # IMPORTANT start  无监督思想
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        # IMPORTANT end

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print("Collecting the experience")
for ep in range(400):
    s = env.reset()
    ep_reward = 0
    while 1:
        env.render()
        action = dqn.choose_action(s)

        s_, r, done, info = env.step(action)

        x, x_dot, theta, theta_dot = s_
        # modify the reward
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ep_reward += r
        dqn.store_transition(s, r, action, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Episode:', ep, 'Reward:', round(ep_reward, 2))

        if done:
            break

        s = s_



