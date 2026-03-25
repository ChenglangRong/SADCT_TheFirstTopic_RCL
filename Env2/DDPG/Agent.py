import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from Network import DDPGActor, DDPGCritic


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), \
            np.array(next_state), np.array(done), np.array(mask)

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, device):
        # 策略网络
        self.actor = DDPGActor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = DDPGActor(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = DDPGCritic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = DDPGCritic(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state).cpu().detach().numpy()[0]

        # 应用动作掩码（强化版）
        masked_probs = probs * np.array(mask, dtype=np.float32)  # 转为float32避免类型问题

        # 处理所有动作都被屏蔽的极端情况（强制至少一个合法动作）
        if masked_probs.sum() < 1e-8:
            # 若所有动作都非法，强制mask为全1（环境应避免此情况，但留作保险）
            masked_probs = np.ones_like(probs)
            print("警告：所有动作被屏蔽，强制允许所有动作")

        # 归一化概率
        masked_probs = masked_probs / masked_probs.sum()

        # 按概率采样动作
        action = np.random.choice(self.action_dim, p=masked_probs)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # 采样
        state, action, reward, next_state, done, mask = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 目标动作选择（带掩码，强化版）
        next_probs = self.target_actor(next_state)
        masked_next_probs = next_probs * mask  # 过滤非法动作

        # 重新归一化（处理可能的数值误差）
        sum_next_probs = masked_next_probs.sum(dim=1, keepdim=True)
        sum_next_probs = torch.where(sum_next_probs == 0, torch.ones_like(sum_next_probs), sum_next_probs)
        masked_next_probs = masked_next_probs / sum_next_probs

        # 选择概率最高的合法动作
        next_action = torch.argmax(masked_next_probs, dim=1).view(-1, 1)

        # 目标Q值计算
        next_q_values = self.target_critic(next_state, F.one_hot(next_action.squeeze(), self.action_dim).float())
        q_targets = reward + self.gamma * (1 - done) * next_q_values

        # 当前Q值
        action_onehot = F.one_hot(action, self.action_dim).float()
        # 应用mask到one-hot向量（确保非法动作的one-hot为0）
        action_onehot = action_onehot * mask
        q_values = self.critic(state, action_onehot)

        #  Critic更新
        critic_loss = F.mse_loss(q_values, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新
        actor_probs = self.actor(state)  # 原始概率分布

        # 关键修改：应用mask并重新归一化，确保非法动作概率为0
        masked_actor_probs = actor_probs * mask  # 过滤非法动作
        # 防止除零错误（若所有动作都非法，mask会处理为全1，此处只是双重保险）
        sum_probs = masked_actor_probs.sum(dim=1, keepdim=True)
        sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
        masked_actor_probs = masked_actor_probs / sum_probs  # 重新归一化

        # 计算当前策略下的动作分布熵（仅基于合法动作）
        entropy = -torch.sum(masked_actor_probs * torch.log(masked_actor_probs + 1e-8), dim=1).mean()
        # 最大熵策略优化（使用过滤后的概率）
        q_values_actor = self.critic(state, masked_actor_probs)
        actor_loss = -torch.mean(q_values_actor) - 0.001 * entropy  # 熵正则化

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()