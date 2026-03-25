import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """基础经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask = zip(*transitions)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done), np.array(mask))

    def size(self):
        return len(self.buffer)


class TD3:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=100000,
                 batch_size=64, policy_delay=2, device='cpu'):
        # 策略网络与目标网络
        self.actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 价值网络与目标网络
        self.critic = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 软更新系数
        self.batch_size = batch_size
        self.policy_delay = policy_delay  # Actor延迟更新步数
        self.action_dim = action_dim
        self.device = device
        self.update_cnt = 0  # 更新计数器

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def take_action(self, state, mask, epsilon=0.1):
        """选择动作：ε-贪婪策略"""
        if random.random() < epsilon:
            # 随机选择有效动作
            valid_actions = np.where(np.array(mask) == 1)[0]
            return np.random.choice(valid_actions) if valid_actions.size > 0 else 0
        else:
            # 策略网络选择动作
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state).cpu().detach().numpy()[0]
            # 过滤无效动作
            probs *= np.array(mask, dtype=np.float32)
            probs /= probs.sum()  # 重新归一化
            return np.random.choice(self.action_dim, p=probs)

    def soft_update(self, net, target_net):
        """软更新目标网络"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        """更新网络参数"""
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        # 采样经验
        state, action, reward, next_state, done, mask = self.replay_buffer.sample(self.batch_size)
        # 转换为张量
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 目标动作（下一状态的最优动作）
        next_action_probs = self.target_actor(next_state)
        next_action = torch.argmax(next_action_probs * mask, dim=1).view(-1, 1)
        next_action_onehot = torch.zeros_like(next_action_probs).scatter(1, next_action, 1)

        # 计算目标Q值（双Critic取最小值）
        target_q1, target_q2 = self.target_critic(next_state, next_action_onehot)
        target_q = torch.min(target_q1, target_q2)
        q_target = reward + self.gamma * (1 - done) * target_q

        # 计算当前Q值
        action_onehot = torch.zeros_like(next_action_probs).scatter(1, action.view(-1, 1), 1)
        current_q1, current_q2 = self.critic(state, action_onehot)

        # 更新Critic
        critic_loss = F.mse_loss(current_q1, q_target.detach()) + F.mse_loss(current_q2, q_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor
        actor_loss = None
        self.update_cnt += 1
        if self.update_cnt % self.policy_delay == 0:
            # 计算Actor损失（最大化Q值）
            action_probs = self.actor(state)
            q1, _ = self.critic(state, action_probs)
            actor_loss = -torch.mean(q1)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)
            self.update_cnt = 0

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None