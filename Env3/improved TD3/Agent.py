import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from Network import TD3Actor, TD3Critic, TD3TargetCritic
from torch.optim import RAdam
from torch.optim.lr_scheduler import CyclicLR

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.pos = 0
        self.time_stamps = np.zeros((capacity,), dtype=np.int32)  # 新增：记录样本时间戳
        self.global_step = 0  # 新增：全局步数计数器，用于时间戳

    # 添加size方法，返回缓冲区当前元素数量
    def size(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, mask, priority=None):
        max_prio = self.priorities.max() if self.memory else 1.0
        if priority is None:
            priority = max_prio
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done, mask))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done, mask)
        self.priorities[self.pos] = priority ** self.alpha
        self.time_stamps[self.pos] = self.global_step  # 更新时间戳
        self.global_step += 1  # 全局步数递增
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios / prios.sum()  # 采样概率

        # 生成并保存采样索引
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 解包样本
        states, actions, rewards, next_states, dones, masks = zip(*samples)

        # 返回8个值（增加indices）
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(masks),
                torch.tensor(weights, dtype=torch.float), indices)

    def update_priorities(self, indices, td_errors):
        # 用self.size()方法和self.capacity属性
        current_alpha = self.alpha * min(1.0, self.size() / self.capacity)
        for idx, error in zip(indices, td_errors):
            # 使用已初始化的time_stamps
            time_factor = 1.0 + 0.01 * self.time_stamps[idx]
            self.priorities[idx] = (abs(error) * time_factor + 1e-6) ** current_alpha


class TD3:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, buffer_size, batch_size, policy_delay=2, lr_decay=0.995, device='cpu'):
        # 策略网络
        self.actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = TD3Actor(state_dim, hidden_dim, action_dim).to(device)

        # 价值网络（双Critic）
        self.critic = TD3Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = TD3TargetCritic(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = RAdam(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=1e-5
        )
        self.critic_optimizer = RAdam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=1e-5
        )

        # 循环学习率调度器
        self.actor_lr_scheduler = CyclicLR(
            self.actor_optimizer,
            base_lr=actor_lr * 0.1,
            max_lr=actor_lr,
            step_size_up=2000
        )
        self.critic_lr_scheduler = CyclicLR(
            self.critic_optimizer,
            base_lr=critic_lr * 0.1,
            max_lr=critic_lr,
            step_size_up=2000
        )

        # 超参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.batch_size = batch_size
        self.policy_delay = policy_delay  # Actor延迟更新步数
        self.action_dim = action_dim
        self.device = device
        self.update_cnt = 0  # 计数器，用于控制Actor更新

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)


    # 在Agent.py的TD3类中添加以下方法
    def bias_exploration(self, valid_actions):
        """偏向选择未探索过的动作，鼓励探索"""
        # 初始化动作探索计数（如果不存在）
        if not hasattr(self, 'action_explore_counts'):
            self.action_explore_counts = {action: 0 for action in range(self.action_dim)}

        # 对有效动作按探索次数排序（次数少的优先）
        valid_actions_sorted = sorted(valid_actions, key=lambda x: self.action_explore_counts[x])

        # 给探索少的动作更高的选择概率
        # 这里采用简单的线性概率分布：探索次数越少，权重越高
        weights = [1.0 / (self.action_explore_counts[action] + 1) for action in valid_actions_sorted]
        weights = np.array(weights) / np.sum(weights)  # 归一化

        # 选择动作
        chosen_action = np.random.choice(valid_actions_sorted, p=weights)

        # 更新探索计数
        self.action_explore_counts[chosen_action] += 1

        return chosen_action


    def take_action(self, state, mask, epsilon = 1):

        # 自适应噪声参数（随训练进程减小）
        noise_scale = max(0.01, 0.1 - 0.09 * (self.update_cnt / 100000))

        # 改进epsilon-贪婪策略：基于当前训练进度动态调整
        if random.random() < epsilon:
            valid_actions = np.where(np.array(mask) == 1)[0]
            if len(valid_actions) == 0:
                return 0  # 无有效动作时的保底
            # 偏向选择未探索过的动作
            return self.bias_exploration(valid_actions)

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state).cpu().detach().numpy()[0]

        # 改进动作掩码处理：确保概率归一化稳定性
        masked_probs = probs * np.array(mask, dtype=np.float32)
        sum_probs = masked_probs.sum()

        if sum_probs < 1e-8:
            return np.random.choice(len(probs))  # 极端情况随机选择
        masked_probs /= sum_probs

        print(f"动作概率分布: {masked_probs}")

        # 采用高斯噪声增强探索（适用于离散动作的概率分布）
        noise = np.random.normal(0, noise_scale, size=masked_probs.shape)
        masked_probs = np.clip(masked_probs + noise, 0, 1)
        masked_probs /= masked_probs.sum()  # 重新归一化

        masked_probs_tensor = torch.tensor(masked_probs, dtype=torch.float)
        action_dist = torch.distributions.Categorical(masked_probs_tensor)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        print(f"选择的动作: {action.item()}, 动作对数概率: {log_prob.item()}")

        return np.random.choice(self.action_dim, p=masked_probs)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self):
        if self.replay_buffer.size() < self.batch_size * 10:
            return None, None

            # 采样时获取indices（需修改ReplayBuffer.sample返回indices）
        state, action, reward, next_state, done, mask, weights, indices = self.replay_buffer.sample(self.batch_size)

        # 将权重转换为tensor并移动到设备
        weights = weights.clone().detach().view(-1, 1).to(self.device)

        # 转换其他数据类型（保持不变）
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float).to(self.device)

        # 目标动作选择（保持不变）
        next_probs = self.target_actor(next_state)
        masked_next_probs = next_probs * mask  # 过滤非法动作

        # 重新归一化（保持不变）
        sum_next_probs = masked_next_probs.sum(dim=1, keepdim=True)
        sum_next_probs = torch.where(sum_next_probs == 0, torch.ones_like(sum_next_probs), sum_next_probs)
        masked_next_probs = masked_next_probs / sum_next_probs

        # 选择概率最高的合法动作（保持不变）
        next_action = torch.argmax(masked_next_probs, dim=1).view(-1, 1)
        next_action_onehot = F.one_hot(next_action.squeeze(), self.action_dim).float()

        # 目标Q值计算（取两个目标网络的最小值）（保持不变）
        target_q1, target_q2 = self.target_critic(next_state, next_action_onehot)
        target_q = torch.min(target_q1, target_q2)
        q_targets = reward + self.gamma * (1 - done) * target_q

        # 当前Q值（保持不变）
        action_onehot = F.one_hot(action, self.action_dim).float()
        action_onehot = action_onehot * mask  # 应用mask
        current_q1, current_q2 = self.critic(state, action_onehot)

        # 计算当前Q与目标Q的误差（TD误差）
        td_error1 = current_q1 - q_targets.detach()
        td_error2 = current_q2 - q_targets.detach()
        td_errors = (td_error1.abs() + td_error2.abs()) / 2  # 平均TD误差

        # 用TD误差更新回放池优先级
        self.replay_buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

        # 改进Critic损失计算（添加权重）
        critic_loss = (F.smooth_l1_loss(current_q1, q_targets.detach(), reduction='none') * weights).mean() + \
                      (F.smooth_l1_loss(current_q2, q_targets.detach(), reduction='none') * weights).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor（保持不变）
        self.update_cnt += 1
        actor_loss = None
        if self.update_cnt % self.policy_delay == 0:
            # Actor更新
            actor_probs = self.actor(state)
            masked_actor_probs = actor_probs * mask  # 过滤非法动作

            if self.update_cnt % (self.policy_delay * 100) == 0:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            # 防止除零错误
            sum_probs = masked_actor_probs.sum(dim=1, keepdim=True)
            sum_probs = torch.where(sum_probs == 0, torch.ones_like(sum_probs), sum_probs)
            masked_actor_probs = masked_actor_probs / sum_probs

            # 计算熵正则化
            entropy = -torch.sum(masked_actor_probs * torch.log(masked_actor_probs + 1e-8), dim=1).mean()

            # 使用第一个Critic计算Q值
            q1, _ = self.critic(state, masked_actor_probs)
            actor_loss = -torch.mean(q1) - 0.001 * entropy  # 熵正则化

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)

            self.update_cnt = 0  # 重置计数器

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None