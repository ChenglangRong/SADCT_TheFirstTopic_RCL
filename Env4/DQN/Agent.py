import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from clustertool.PPO.params import actions
from clustertool.SACT_ConcurrentProcessing_ENV1 import Environment
import warnings
from pylab import mpl
from clustertool.DQN_env1.Network import Qnet

# -------------------------------------------------
#   DQN_env1-Agent
# -------------------------------------------------
class DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_end, epsilon_dec, epsilon_dec_steps, target_update, device,ckpt_dir,elect_env_example):
        self.elect_env_example = elect_env_example
        # 动作空间维度
        self.action_dim = action_dim

        # Q网络
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)

        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

        self.gamma = gamma                  # 折扣因子
        self.epsilon = epsilon              # epsilon-贪婪策略，epsilon的初始值
        self.epsilon_end = epsilon_end      # epsilon-贪婪策略，epsilon的终值
        self.epsilon_dec = epsilon_dec      # epsilon-贪婪策略，epsilon的衰减率
        self.epsilon_dec_steps = epsilon_dec_steps   # epsilon-贪婪策略，epsilon进行一次的衰减的步长
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.checkpoint_dir = ckpt_dir
        self.randomaction = False

    def take_action(self, state, is_test):
        if np.random.random() < self.epsilon and not is_test:
            print("<<<<<随机选择动作>>>>>")
            self.randomaction = True
            action = np.random.randint(self.action_dim)
        else:
            self.randomaction = False
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            print(f"当前价值最大的动作：{action}", self.q_net(state))
        return action

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon * self.epsilon_dec
        else:
            self.epsilon = self.epsilon_end


    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值

        max_next_q_values = self.target_q_net(next_states).max(1)[0].view( -1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()         # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.epsilon_dec_steps == 0 :
            self.decrement_epsilon()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        return dqn_loss.item()


    def save_models(self, episode):
        self.q_net.save_checkpoint(self.checkpoint_dir + 'Q_eval_env_{}/DQN_q_eval_{}.pth'.format(self.elect_env_example,episode))
        print('Saving Q_eval network successfully!')
        self.target_q_net.save_checkpoint(self.checkpoint_dir + 'Q_target_env_{}/DQN_Q_target_{}.pth'.format(self.elect_env_example,episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_net.load_checkpoint(self.checkpoint_dir + 'Q_eval_env_{}/DQN_q_eval_{}.pth'.format(self.elect_env_example,episode))
        print('Loading Q_eval network successfully!')
        self.target_q_net.load_checkpoint(self.checkpoint_dir + 'Q_target_env_{}/DQN_Q_target_{}.pth'.format(self.elect_env_example,episode))
        print('Loading Q_target network successfully!')