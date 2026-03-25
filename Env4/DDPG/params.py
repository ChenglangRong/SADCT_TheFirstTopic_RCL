# ========================训练超参数=======================
import argparse

parser = argparse.ArgumentParser()
actor_lr = 1e-3     # 策略网络学习率
critic_lr = 5e-3    # 价值网络学习率
num_episodes = 2000   # 回合数
hidden_dim = 128       # 隐藏层神经元数量
gamma = 0.95           # 折扣因子
tau = 0.001            # 软更新系数
buffer_size = 10000    # 回放池大小
batch_size = 256       # 批处理大小
parser_ddpg = argparse.ArgumentParser()
parser_ddpg.add_argument('--actor_lr', type=float, default=actor_lr)
parser_ddpg.add_argument('--critic_lr', type=float, default=critic_lr)
parser_ddpg.add_argument('--num_episodes', type=int, default=num_episodes)
parser_ddpg.add_argument('--hidden_dim', type=int, default=hidden_dim)
parser_ddpg.add_argument('--gamma', type=float, default=gamma)
parser_ddpg.add_argument('--tau', type=float, default=tau)
parser_ddpg.add_argument('--buffer_size', type=int, default=buffer_size)
parser_ddpg.add_argument('--batch_size', type=int, default=batch_size)
parser_ddpg.add_argument('--ckpt_dir', type=str, default='checkpoints/')
parser_ddpg.add_argument('--image_dir', type=str, default='images/')
parser_ddpg.add_argument('--data_dir', type=str, default='data/')
args_DDPG = parser_ddpg.parse_args()



#===================================================================案例6
robot_actions = ["MT", "WT", "UT", "CT", "LT", "IDLE"]  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
modules_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
steps_list = ["LL", "PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "LL"]
process_residency_time_dict = {"LL": [0, 0], "PM1": [100, 20], "PM2": [100, 20], "PM3": [110, 20], "PM4": [110, 20],
                                "PM5": [115, 20], "PM6": [115, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [100, 20], [110, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 100, "PM2": 100, "PM3": 110, "PM4": 110, "PM5": 115, "PM6": 115}
process_time_list = [0, 100, 100, 110, 110, 115, 115]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2],    # 0 1
        [2, 3], [3, 4], [4, 0],   # 类型1路径: LL->PM1->PM2->PM3->PM4->LL   2 3 4
        [2, 5], [5, 6], [6, 0],   # 类型2路径: LL->PM1->PM2->PM5->PM6->LL   5 6 7
]
action_dim = len(actions)
wafer_num = 10  # 总晶圆数量
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
process_residency_time_dict = {"LL": [0, 0], "PM1": [100, 20], "PM2": [100, 20], "PM3": [110, 20], "PM4": [110, 20],
                                "PM5": [115, 20], "PM6": [115, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [100, 20], [110, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 100, "PM2": 100, "PM3": 110, "PM4": 110, "PM5": 115, "PM6": 115}
process_time_list = [0, 100, 100, 110, 110, 115, 115]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2],    # 0 1
        [2, 3], [3, 4], [4, 0],   # 类型1路径: LL->PM1->PM2->PM3->PM4->LL   2 3 4
        [2, 5], [5, 6], [6, 0],   # 类型2路径: LL->PM1->PM2->PM5->PM6->LL   5 6 7
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
process_residency_time_dict = {"LL": [0, 0], "PM1": [100, 20], "PM2": [100, 20], "PM3": [110, 20], "PM4": [110, 20],
                                "PM5": [115, 20], "PM6": [115, 20]}
process_residency_time_list = [[0, 0], [90, 20], [120, 20], [120, 20], [100, 20], [110, 20], [100, 20]]
max_stay_time_dict = {"LL": 0, "PM1": 20, "PM2": 20, "PM3": 20, "PM4": 20, "PM5": 20, "PM6": 20}
max_stay_time_list = [0, 20, 20, 20, 20, 20, 20]
process_time_dict = {"LL": 0, "PM1": 100, "PM2": 100, "PM3": 110, "PM4": 110, "PM5": 115, "PM6": 115}
process_time_list = [0, 100, 100, 110, 110, 115, 115]
wait_time_dict = {"LL": 0, "PM1": 0, "PM2": 0, "PM3": 0, "PM4": 0, "PM5": 0, "PM6": 38}
wait_time_list = [0, 0, 0, 0, 0, 0, 38]
unload_time_LL = 10
work_time = 5
move_time = 2
# 动作空间
actions = [
        [0, 1], [1, 2],    # 0 1
        [2, 3], [3, 4], [4, 0],   # 类型1路径: LL->PM1->PM2->PM3->PM4->LL   2 3 4
        [2, 5], [5, 6], [6, 0],   # 类型2路径: LL->PM1->PM2->PM5->PM6->LL   5 6 7
]
action_dim = len(actions)
wafer_num = 10  # 总晶圆数量
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
