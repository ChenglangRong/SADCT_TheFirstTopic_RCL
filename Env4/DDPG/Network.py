import torch
import torch.nn as nn
import torch.nn.functional as F

# DDPG Actor网络（离散动作）
class DDPGActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.temperature = 2.0  # 温度系数控制探索性

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # 限制logits范围
        logits = torch.clamp(logits, min=0.01, max=0.9)
        # 应用温度系数软化分布
        probs = F.softmax(logits / self.temperature, dim=1)
        return probs

# DDPG Critic网络
class DDPGCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DDPGCritic, self).__init__()
        # 状态输入分支
        self.fc1 = nn.Linear(state_dim, hidden_dim//2)
        # 动作输入分支
        self.fc2 = nn.Linear(action_dim, hidden_dim//2)
        # 合并层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(action))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)