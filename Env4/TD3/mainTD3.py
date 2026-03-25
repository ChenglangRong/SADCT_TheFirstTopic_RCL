from Agent import TD3
import torch
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from clustertool.SACT_ENV4 import Environment
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import utils
from utils import RunningMeanStd

class TD3_runner():
    def __init__(self, args_TD3, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        self.args_TD3 = args_TD3  # 保存参数引用
        self.actor_lr = args_TD3.actor_lr
        self.critic_lr = args_TD3.critic_lr
        self.num_episodes = args_TD3.num_episodes
        self.hidden_dim = args_TD3.hidden_dim
        self.gamma = args_TD3.gamma
        self.tau = args_TD3.tau
        self.buffer_size = args_TD3.buffer_size
        self.batch_size = args_TD3.batch_size
        self.policy_delay = args_TD3.policy_delay
        self.epsilon_decay = args_TD3.epsilon_decay
        self.epsilon_final = args_TD3.epsilon_final
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env

        # 创建环境
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        # 再初始化状态归一化器
        self.state_normalizer = RunningMeanStd(shape=self.state_dim)

        # 初始化智能体
        self.agent = TD3(
            self.state_dim, self.hidden_dim, self.action_dim,
            self.actor_lr, self.critic_lr, self.gamma, self.tau,
            self.buffer_size, self.batch_size, self.policy_delay, self.device
        )

        # 创建保存目录
        utils.create_directory(args_TD3.ckpt_dir,
                               sub_dirs=[f'actor_env_{elect_env_example}',
                                         f'critic_env_{elect_env_example}'])

        # 记录数据
        self.reward_list = []
        self.makespan_list = []
        self.data = {
            'episode': [], 'reward': [], 'makespan': [], 'fail': []
        }

    def run(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # 记录损失值，便于监控训练状态
        self.actor_loss_list = []
        self.critic_loss_list = []

        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            print(f"\n第{i_episode + 1}回合====================================")
            step_count = 0  # 在每个回合开始时初始化计数器

            epsilon = self.epsilon_final + (1.0 - self.epsilon_final) * np.exp(-1. * i_episode / (self.epsilon_decay / 2))

            while not done:
                # 状态归一化
                normalized_state = self.state_normalizer.normalize(state)
                mask = self.env.get_mask()
                action = self.agent.take_action(normalized_state, mask, epsilon)
                next_state, reward, done = self.env.step(action)
                # 存储经验时添加时间戳或优先级（简易优先级：奖励绝对值）
                priority = abs(reward) + 1e-6  # 避免为0
                self.agent.replay_buffer.add(state, action, reward, next_state, done, mask, priority)
                state = next_state
                episode_return += reward
                step_count += 1 # 每步递增计数器

                # 使用本地维护的step_count替代self.env.step_count
                if self.agent.replay_buffer.size() >= self.batch_size * 10:
                    if step_count % 2 == 0:  # 每2步更新一次
                        critic_loss, actor_loss = self.agent.update()
                        if critic_loss is not None:
                            self.critic_loss_list.append(critic_loss)
                        if actor_loss is not None:
                            self.actor_loss_list.append(actor_loss)
                            
                # 更新归一化器
                self.state_normalizer.update(state)

            self.reward_list.append(episode_return)
            self.data['episode'].append(i_episode)
            self.data['reward'].append(episode_return)
            self.data['makespan'].append(self.env.env.now)
            self.data['fail'].append(self.env.fail_flag)

            print(f"奖励: {episode_return}, 完工时间: {self.env.env.now}, Epsilon: {epsilon:.3f}")
            if not self.env.fail_flag:
                self.makespan_list.append(self.env.env.now)

            # 定期保存模型（每100回合）
            if (i_episode + 1) % 100 == 0:
                torch.save(self.agent.actor.state_dict(),
                           f"{self.args_TD3.ckpt_dir}/actor_env_{self.elect_env_example}/episode_{i_episode}.pth")

        return self.reward_list, self.makespan_list, self.data


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