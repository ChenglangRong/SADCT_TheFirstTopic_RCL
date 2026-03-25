from Agent import TD3
import torch
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from clustertool.SACT_ENV1 import Environment
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import utils
from utils import RunningMeanStd

class TD3Runner:
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_episodes=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Environment()  # 初始化环境（根据实际环境参数调整）
        self.num_episodes = num_episodes

        # 初始化智能体
        self.agent = TD3(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            device=self.device
        )

        # 记录训练数据
        self.rewards = []
        self.makespans = []

    def run(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                mask = self.env.get_mask()  # 获取动作掩码
                action = self.agent.take_action(state, mask, epsilon=0.1)
                next_state, reward, done = self.env.step(action)
                # 存储经验
                self.agent.replay_buffer.add(state, action, reward, next_state, done, mask)
                state = next_state
                total_reward += reward

                # 更新网络
                if self.agent.replay_buffer.size() > 1000:
                    self.agent.update()

            self.rewards.append(total_reward)
            self.makespans.append(self.env.env.now)  # 记录完工时间
            print(f"Episode {episode+1}: Reward={total_reward:.2f}, Makespan={self.env.env.now}")

        return self.rewards, self.makespans


def set_env(elect_env_example):
    from params import args_TD3  # 需要在params.py中添加TD3参数

    if elect_env_example == 6:
        from params import args_6 as args_env
    elif elect_env_example == 7:
        from params import args_7 as args_env
    elif elect_env_example == 8:
        from params import args_8 as args_env
    else:
        print("案例选择错误")
        exit(1)
    return TD3_runner(args_TD3, args_env, elect_env_example)


def main():
    from params import args_TD3
    elect_env_example = 7  # 选择案例
    runner = set_env(elect_env_example)

    # 创建保存目录
    utils.create_directory(args_TD3.image_dir, [f'TD3_env_{elect_env_example}'])
    utils.create_directory(args_TD3.data_dir, [f'TD3_env_{elect_env_example}'])

    # 运行训练
    reward_list, makespan_list, data = runner.run()

    # 保存结果
    utils.save_data(data,
                    f"{args_TD3.data_dir}/TD3_env_{elect_env_example}/TD3_data.xlsx")

    # 绘制奖励曲线
    episodes = range(len(reward_list))
    utils.plot_method(episodes, reward_list,
                      'Reward', 'reward',
                      f"{args_TD3.image_dir}/TD3_env_{elect_env_example}/reward.png")

    # 绘制移动平均奖励
    avg_rewards = utils.moving_average(reward_list, 400)
    utils.plot_method(range(len(avg_rewards)), avg_rewards,
                      'Moving cumulated episode reward', 'reward',
                      f"{args_TD3.image_dir}/TD3_env_{elect_env_example}/avg_reward.png")

    # 绘制makespan散点图
    if makespan_list:  # 确保有有效的makespan数据
        # 获取成功回合的索引
        episodes_success = [i for i, fail in enumerate(data['fail']) if not fail]

        # 绘制散点图
        utils.scatter_method(episodes_success,makespan_list,'Makespan','makespan',
            f"{args_TD3.image_dir}/TD3_env_{elect_env_example}/TD3_makespan_env_{elect_env_example}.png"
        )


if __name__ == "__main__":
    main()