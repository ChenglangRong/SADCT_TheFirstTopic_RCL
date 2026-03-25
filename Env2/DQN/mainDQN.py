import random
import warnings
import numpy as np
import torch
from pylab import mpl
from clustertool.DQN_env2.Agent import DQN
from clustertool.DQN_env2.ReplayBuffer import ReplayBuffer
from clustertool.DQN_env2.utils import moving_average, create_directory, save_data, plot_method,scatter_method
from clustertool.SACT_ENV2 import Environment

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

warnings.filterwarnings("ignore",category=DeprecationWarning)


"""
DQN_env2 runner
"""
class DQN_runner():
    def __init__(self, args_DQN, args_env, elect_env_example):
        self.elect_env_example = elect_env_example
        # ========训练超参数=======
        self.learn_rate = args_DQN.learn_rate                # 学习率
        self.num_episodes = args_DQN.num_episodes            # 回合数
        self.hidden_dim = args_DQN.hidden_dim                # 回合数
        self.gamma = args_DQN.gamma                          # 折扣因子
        self.epsilon = args_DQN.epsilon                      # 初始epsilon
        self.epsilon_end = args_DQN.epsilon_end              # 最终epsilon
        self.epsilon_dec = args_DQN.epsilon_dec              # epsilon衰减率
        self.epsilon_dec_steps = args_DQN.epsilon_dec_steps  # epsilon频率
        self.target_update = args_DQN.target_update          # 更新目标网络的episode间隔
        self.buffer_size = args_DQN.buffer_size              # 经验回放池的大小
        self.minimal_size = args_DQN.minimal_size            # 初始时经验回放池为空，只有收集到一定数量的数据后，才开始进行Q网络的训练
        self.batch_size = args_DQN.batch_size                # 进行Q网络训练的数据采样的大小
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.env_args = args_env
        # ========创建环境=======
        self.env = Environment(args_env, args_env.wafer_num)
        self.state_dim = self.env.state_dim  # 状态空间的维度
        self.action_dim = self.env.action_dim  # 动作空间的维度

        self.replay_buffer = ReplayBuffer(self.buffer_size)  # 实例经验回放池

        self.agent = DQN(self.state_dim, self.hidden_dim, self.action_dim, self.learn_rate,
                         self.gamma, self.epsilon, self.epsilon_end, self.epsilon_dec,self.epsilon_dec_steps,
                         self.target_update, self.device,args_DQN.ckpt_dir,self.elect_env_example)
        self.agent.env = self.env  # 将环境实例传递给Agent

        create_directory(args_DQN.ckpt_dir, sub_dirs=['Q_eval_env_'+str(elect_env_example), 'Q_target_env_'+str(elect_env_example)])

        self.reward_list = []
        self.makespan_list = []
        self.reward_tuple = []
        self.loss_tuple = []
        self.loss_list = []
        self.epsilon_list = []
        self.data = {
            'episode': [],
            'epsilon': [],
            'reward': [],
            'loss': [],
            'makespan': [],
            'randomaction':[],
            'fail':[]
        }

    def run(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        for i_episode in range(int(self.num_episodes)):
            episode_return = 0
            total_reward = 0
            total_loss = 0
            loss = 0
            state = self.env.reset()
            done = False
            print(
                f"\n\n第{i_episode + 1}个回合==============================================================================================================================================")

            while not done:
                print(f"\nEP:{i_episode}***********epsilon:", self.agent.epsilon)
                action = self.agent.take_action(state, False)  # 获取动作
                next_state, reward, done = self.env.step(action)  # 执行动作，返回下一个状态、奖励和标记
                self.replay_buffer.add(state, action, reward, next_state, done)  # 存入经验回放池
                state = next_state  # 更新状态
                episode_return += reward  # 更新奖励
                total_reward += reward
                loss = 0
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if self.replay_buffer.size() > self.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)  # 取样
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    loss = self.agent.update(transition_dict)
                    total_loss += loss

            self.loss_list.append(total_loss)
            self.loss_tuple.append((i_episode, total_loss))

            self.reward_list.append(episode_return)
            self.reward_tuple.append((i_episode, episode_return))

            self.epsilon_list.append(self.agent.epsilon)

            self.data['episode'].append(i_episode)
            self.data['epsilon'].append(self.agent.epsilon)
            self.data['reward'].append(episode_return)
            self.data['loss'].append(total_loss)
            self.data['makespan'].append(self.env.env.now)
            self.data['randomaction'].append(self.agent.randomaction)
            self.data['fail'].append(self.env.fail_flag)

            print('EP:{} reward:{} loss:{} epsilon:{} makespan:{}'.format(i_episode + 1, episode_return, total_loss,
                                                                          self.agent.epsilon, self.env.env.now))
            if not self.env.fail_flag:
                self.makespan_list.append(self.env.env.now)

        return self.reward_list, self.reward_tuple, self.loss_list, self.loss_tuple, self.epsilon_list, self.makespan_list, self.data


def set_env(elect_env_example):

    from params import args_DQN

    if elect_env_example == 6:
        from params import args_6 as args_env
        SACT = DQN_runner(args_DQN,  args_env, elect_env_example)
    elif elect_env_example == 7:
        from params import args_7 as args_env
        SACT = DQN_runner(args_DQN, args_env, elect_env_example)
    elif elect_env_example == 8:
        from params import args_8 as args_env
        SACT = DQN_runner(args_DQN, args_env, elect_env_example)
    else:
        print("选择案例标号出错")
        exit(1)
    return SACT


def main():
    reward_list = []
    makespan_list = []
    reward_tuple = []
    loss_tuple = []
    loss_list = []
    epsilon_list = []
    data = {}
    from params import args_DQN
    elect_env_example = 7                             # 这里修改不同的案例
    SACT = set_env(elect_env_example)

    create_directory(args_DQN.image_dir,['SACT_env_'+str(elect_env_example)])
    create_directory(args_DQN.data_dir,['SACT_env_'+str(elect_env_example)])

    # ---------运行DQN-runner---------
    reward_list, reward_tuple, loss_list, loss_tuple, epsilon_list, makespan_list, data = SACT.run()

    # ---------保存data[包括每个回合的reward,loss和epsilon]---------
    save_data(data, args_DQN.data_dir + "/SACT_env_{}/SACT_data_env_{}.xlsx".format(str(elect_env_example),
                                                                                    str(elect_env_example)))
    episodes = [i for i in range(args_DQN.num_episodes)]
    plot_method(episodes, reward_list, 'Reward', 'reward', args_DQN.image_dir+"/SACT_env_{}/SACT_reward_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    plot_method(episodes, moving_average(reward_list, 199), 'Moving cumulated episode reward', 'reward', args_DQN.image_dir+"/SACT_env_{}/SACT_avg_reward_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    plot_method(episodes, loss_list, 'Loss', 'loss', args_DQN.image_dir+"/SACT_env_{}/SACT_loss_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    plot_method(episodes, moving_average(loss_list, 199), 'Moving cumulated episode loss', 'loss', args_DQN.image_dir+"/SACT_env_{}/SACT_avg_loss_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    plot_method(episodes, epsilon_list, 'Epsilon', 'epsilon', args_DQN.image_dir+"/SACT_env_{}/SACT_epsilon_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    # plot_method(episodes, moving_average(makespan_list, 199), 'Moving cumulated episode makespan', 'makespan', args_DQN.image_dir+"/NCDP_env_{}/NCDP_makespan_env_{}.png".format(str(elect_env_example),str(elect_env_example)))
    episodes_success = [i for i in range(len(makespan_list))]
    scatter_method(episodes_success, makespan_list, 'Makespan', 'makespan', args_DQN.image_dir+"/SACT_env_{}/SACT_makespan_env_{}.png".format(str(elect_env_example),str(elect_env_example)))



if __name__ == '__main__':
    main()











