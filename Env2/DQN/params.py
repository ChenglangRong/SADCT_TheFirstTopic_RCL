import argparse

# ========================训练超参数=======================
parser = argparse.ArgumentParser()
learn_rate = 0.003     # 学习率
num_episodes = 50000   # 回合数
hidden_dim = 128       # 隐藏层神经元数量
gamma = 0.99           # 折扣因子
epsilon = 0.5          # 初始epsilon
epsilon_end = 0.0001    # 最终epsilon
epsilon_dec = 0.98
epsilon_dec_steps = 1000
target_update = 100     # 更新目标网络的episode间隔
buffer_size = 1500      # 经验回放池的大小
minimal_size = 200      # 初始时经验回放池为空，只有收集到一定数量的数据后，才开始进行Q网络的训练
batch_size = 128        # 进行Q网络训练的数据采样的大小
parser.add_argument('--learn_rate', default=learn_rate, type=float)
parser.add_argument('--num_episodes', default=num_episodes, type=int)
parser.add_argument('--hidden_dim', default=hidden_dim, type=int)
parser.add_argument('--gamma', default=gamma, type=float)
parser.add_argument('--epsilon', default=epsilon, type=float)
parser.add_argument('--epsilon_end', default=epsilon_end, type=float)
parser.add_argument('--epsilon_dec', default=epsilon_dec, type=float)
parser.add_argument('--epsilon_dec_steps', default=epsilon_dec_steps, type=int)
parser.add_argument('--target_update', default=target_update, type=int)
parser.add_argument('--buffer_size', default=buffer_size, type=int)
parser.add_argument('--minimal_size', default=minimal_size, type=int)
parser.add_argument('--batch_size', default=batch_size, type=int)
parser.add_argument('--ckpt_dir', type=str, default='./SACT_checkpoints/DQN_env1/')
parser.add_argument('--image_dir', type=str, default='./SACT_output_images/')
parser.add_argument('--data_dir', type=str, default='./SACT_output_data/')
args_DQN = parser.parse_args()





#===================================================================案例6
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
steps_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LL"]
process_residency_time_dict = {"LL": [0, 0], "PM1": [90, 20], "PM2": [120, 20], "PM3": [120, 20], "PM4": [90, 20],
                               "PM5": [100, 20], "PM6": [100, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [90, 20], [100, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 90, "PM2": 120, "PM3": 120, "PM4": 90, "PM5": 100, "PM6": 100}
process_time_list = [0, 90, 120, 120, 90, 100, 100]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 0],  # 类型1路径: LL→PM1→PM2//PM3→PM4→LL    0 1 2 3 4 5
        [0, 5], [5, 6], [6, 0],  # 类型2路径: LL→PM5→PM6→LL   6 7 8
]
action_dim = len(actions)

wafer_num = 4  # 总晶圆数量
# 两种晶圆各占一半
wafer_type_distribution = [0.5, 0.5]  # 类型1:类型2 = 1:1

# params for cluster-tool:
parser_6 = argparse.ArgumentParser()
parser_6.add_argument('--robot_actions', default=robot_actions, type=list)
parser_6.add_argument('--modules_list', default=modules_list, type=list)
parser_6.add_argument('--steps_list', default=steps_list, type=list)
parser_6.add_argument('--process_residency_time_dict', default=process_residency_time_dict, type=dict)
parser_6.add_argument('--process_residency_time_list', default=process_residency_time_list, type=list)
parser_6.add_argument('--max_stay_time_dict', default=max_stay_time_dict, type=dict)
parser_6.add_argument('--max_stay_time_list', default=max_stay_time_list, type=list)
parser_6.add_argument('--process_time_dict', default=process_time_dict, type=dict)
parser_6.add_argument('--process_time_list', default=process_time_list, type=list)
parser_6.add_argument('--wait_time_dict', default=wait_time_dict, type=dict)
parser_6.add_argument('--wait_time_list', default=wait_time_list, type=list)
parser_6.add_argument('--unload_time_LL', default=unload_time_LL, type=int)
parser_6.add_argument('--work_time', default=work_time, type=int)
parser_6.add_argument('--move_time', default=move_time, type=int)
parser_6.add_argument('--actions', default=actions, type=list)
parser_6.add_argument('--action_dim', default=action_dim, type=int)
parser_6.add_argument('--wafer_num', default=wafer_num, type=int)
args_6 = parser_6.parse_args()



#===================================================================案例7
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
steps_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LL"]
process_residency_time_dict = {"LL": [0, 0], "PM1": [90, 20], "PM2": [120, 20], "PM3": [120, 20], "PM4": [90, 20],
                               "PM5": [100, 20], "PM6": [100, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [90, 20], [100, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 90, "PM2": 120, "PM3": 120, "PM4": 90, "PM5": 100, "PM6": 100}
process_time_list = [0, 90, 120, 120, 90, 100, 100]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 0],  # 类型1路径: LL→PM1→PM2//PM3→PM4→LL    0 1 2 3 4 5
        [0, 5], [5, 6], [6, 0],  # 类型2路径: LL→PM5→PM6→LL   6 7 8
]
action_dim = len(actions)

wafer_num = 10  # 总晶圆数量
# 两种晶圆各占一半
wafer_type_distribution = [0.5, 0.5]  # 类型1:类型2 = 1:1

# params for cluster-tool:
parser_7 = argparse.ArgumentParser()
parser_7.add_argument('--robot_actions', default=robot_actions, type=list)
parser_7.add_argument('--modules_list', default=modules_list, type=list)
parser_7.add_argument('--steps_list', default=steps_list, type=list)
parser_7.add_argument('--process_residency_time_dict', default=process_residency_time_dict, type=dict)
parser_7.add_argument('--process_residency_time_list', default=process_residency_time_list, type=list)
parser_7.add_argument('--max_stay_time_dict', default=max_stay_time_dict, type=dict)
parser_7.add_argument('--max_stay_time_list', default=max_stay_time_list, type=list)
parser_7.add_argument('--process_time_dict', default=process_time_dict, type=dict)
parser_7.add_argument('--process_time_list', default=process_time_list, type=list)
parser_7.add_argument('--wait_time_dict', default=wait_time_dict, type=dict)
parser_7.add_argument('--wait_time_list', default=wait_time_list, type=list)
parser_7.add_argument('--unload_time_LL', default=unload_time_LL, type=int)
parser_7.add_argument('--work_time', default=work_time, type=int)
parser_7.add_argument('--move_time', default=move_time, type=int)
parser_7.add_argument('--actions', default=actions, type=list)
parser_7.add_argument('--action_dim', default=action_dim, type=int)
parser_7.add_argument('--wafer_num', default=wafer_num, type=int)
args_7 = parser_7.parse_args()



#===================================================================案例8
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
steps_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LL"]
process_residency_time_dict = {"LL": [0, 0], "PM1": [90, 20], "PM2": [120, 20], "PM3": [120, 20], "PM4": [90, 20],
                               "PM5": [100, 20], "PM6": [100, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [90, 20], [100, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 90, "PM2": 120, "PM3": 120, "PM4": 90, "PM5": 100, "PM6": 100}
process_time_list = [0, 90, 120, 120, 90, 100, 100]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 0],  # 类型1路径: LL→PM1→PM2//PM3→PM4→LL    0 1 2 3 4 5
        [0, 5], [5, 6], [6, 0],  # 类型2路径: LL→PM5→PM6→LL   6 7 8
]
action_dim = len(actions)

wafer_num = 20  # 总晶圆数量
# 两种晶圆各占一半
wafer_type_distribution = [0.5, 0.5]  # 类型1:类型2 = 1:1

# params for cluster-tool:
parser_8 = argparse.ArgumentParser()
parser_8.add_argument('--robot_actions', default=robot_actions, type=list)
parser_8.add_argument('--modules_list', default=modules_list, type=list)
parser_8.add_argument('--steps_list', default=steps_list, type=list)
parser_8.add_argument('--process_residency_time_dict', default=process_residency_time_dict, type=dict)
parser_8.add_argument('--process_residency_time_list', default=process_residency_time_list, type=list)
parser_8.add_argument('--max_stay_time_dict', default=max_stay_time_dict, type=dict)
parser_8.add_argument('--max_stay_time_list', default=max_stay_time_list, type=list)
parser_8.add_argument('--process_time_dict', default=process_time_dict, type=dict)
parser_8.add_argument('--process_time_list', default=process_time_list, type=list)
parser_8.add_argument('--wait_time_dict', default=wait_time_dict, type=dict)
parser_8.add_argument('--wait_time_list', default=wait_time_list, type=list)
parser_8.add_argument('--unload_time_LL', default=unload_time_LL, type=int)
parser_8.add_argument('--work_time', default=work_time, type=int)
parser_8.add_argument('--move_time', default=move_time, type=int)
parser_8.add_argument('--actions', default=actions, type=list)
parser_8.add_argument('--action_dim', default=action_dim, type=int)
parser_8.add_argument('--wafer_num', default=wafer_num, type=int)
args_8 = parser_8.parse_args()
