import torch
import torch.nn as nn
import torch.nn.functional as F


# TD3 Actor网络（离散动作）
class TD3Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(TD3Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 新增一层隐藏层
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.temperature = nn.Parameter(torch.tensor(2.0))  # 改为可学习参数
        self.dropout = nn.Dropout(0.1)  # 添加dropout减少过拟合

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        logits = torch.clamp(logits, min=0.01, max=0.9)  # 保持输出范围
        # 动态温度系数（限制在合理范围）
        temp = torch.clamp(self.temperature, 0.1, 2.0)
        probs = F.softmax(logits / temp, dim=1)
        return probs


# TD3 Critic网络（双网络结构）
class TD3Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(TD3Critic, self).__init__()
        # 第一个Q网络（带残差连接）
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.residual1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 第一个残差连接，命名为residual1
        self.fc4 = nn.Linear(hidden_dim, 1)

        # 第二个Q网络
        self.fc5 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.residual2 = nn.Linear(state_dim + action_dim, hidden_dim)  # 第二个残差连接，命名为residual2（修复此处）
        self.fc8 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)  # 添加dropout

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        residual1 = self.residual1(sa)  # 使用residual1

        # Q1网络
        q1 = F.relu(self.bn1(self.fc1(sa)))
        q1 = self.dropout(q1)
        q1 = F.relu(self.bn2(self.fc2(q1)))
        q1 = F.relu(self.fc3(q1) + residual1)  # 残差连接
        q1 = self.fc4(q1)

        # Q2网络
        residual2 = self.residual2(sa)  # 使用residual2
        q2 = F.relu(self.bn5(self.fc5(sa)))
        q2 = self.dropout(q2)
        q2 = F.relu(self.bn6(self.fc6(q2)))
        q2 = F.relu(self.fc7(q2) + residual2)  # 残差连接
        q2 = self.fc8(q2)

        q1 = torch.clamp(q1, -2500.0, 3000.0)
        q2 = torch.clamp(q2, -2500.0, 3000.0)

        return q1, q2


# 目标Critic网络（用于计算目标Q值）
class TD3TargetCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(TD3TargetCritic, self).__init__()
        # 第一个目标Q网络（带残差连接和批归一化）
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批归一化层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # 添加批归一化层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.residual1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 残差连接
        self.fc4 = nn.Linear(hidden_dim, 1)

        # 第二个目标Q网络（带残差连接和批归一化）
        self.fc5 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)  # 添加批归一化层
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.bn6 = nn.BatchNorm1d(hidden_dim)  # 添加批归一化层
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.residual2 = nn.Linear(state_dim + action_dim, hidden_dim)  # 残差连接（注意变量名不能重复）
        self.fc8 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        residual1 = self.residual1(sa)  # 第一个残差
        residual2 = self.residual2(sa)  # 第二个残差

        # Q1网络（与TD3Critic结构一致）
        q1 = F.relu(self.bn1(self.fc1(sa)))
        q1 = self.dropout(q1)
        q1 = F.relu(self.bn2(self.fc2(q1)))
        q1 = F.relu(self.fc3(q1) + residual1)  # 残差连接
        q1 = self.fc4(q1)

        # Q2网络（与TD3Critic结构一致）
        q2 = F.relu(self.bn5(self.fc5(sa)))
        q2 = self.dropout(q2)
        q2 = F.relu(self.bn6(self.fc6(q2)))
        q2 = F.relu(self.fc7(q2) + residual2)  # 残差连接
        q2 = self.fc8(q2)

        q1 = torch.clamp(q1, -2500.0, 3000.0)
        q2 = torch.clamp(q2, -2500.0, 3000.0)

        return q1, q2