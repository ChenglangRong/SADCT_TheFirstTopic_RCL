"""

单臂单组合设备调度环境
LL->PM1->PM2->PM3->PM4->LL
LL->PM1->PM2->PM5->PM6->LL

"""
import numpy as np
import simpy
import argparse


# -------------------------------------------------
#   分析器类（无修改）
# -------------------------------------------------
class system_profiler(object):
    def __init__(self, modules_name, module_list, loadlock, robot, tot_wafer):
        self.reward = 0
        self.total_wafer = tot_wafer  # 当前需要加工的晶圆数量
        self.system_state = None
        self.state = list()
        self.modules_name = modules_name  # 名称列表 【PM1,PM2,PM3,PM4,PM5,PM6】
        self.modules_list = module_list  # PM对象列表
        self.entry_wafer = 0
        self.exit_wafer = 0
        self.pre_exit_wafer = 0  # 上一个状态下，完成加工的晶圆数
        self.pre_entry_wafer = 0  # 上一个状态下，待加工的晶圆数
        self.robot = robot  # robot实体
        self.loadlock = loadlock
        self.loadlock_idle_time = 0
        self.robot_current_module = None
        self.robot_wait_time = 0
        self.state_values = list()
        self.pre_state = list()
        self.last_action = None
        self.current_action = None
        self.processing_wafer_count = 0  # 当前系统中需要清空的晶圆数量
        self.wafer_type_count = {1: 0, 2: 0}  # 记录每种类型晶圆的数量
        for name in self.modules_name:  # 各pm当前的状态值
            self.state_values.append(
                {'name': name, 'wafer_count': 0, 'process_time_remaining': 0, 'residency_time_remaining': 0,
                 'wafer_start_time': 0, 'wafer_state': None, 'processed_time': 0, 'residency_time': 0,
                 'wafer_type': None})
            self.pre_state.append(
                {'name': name, 'wafer_count': 0, 'process_time_remaining': 0, 'residency_time_remaining': 0,
                 'wafer_start_time': 0, 'wafer_state': None, 'processed_time': 0, 'residency_time': 0,
                 'wafer_type': None})

    # ======= 更新各PM的状态 =======
    def update_modules_state(self, target, wafer_count, process_time_remaining, residency_time_remaining, pm_state,
                             wafer_start_time, wafer_state, wafer_processed_time, wafer_residency_time,
                             wafer_type=None):
        for item in self.state_values:
            if target == item['name']:
                item['wafer_count'] = wafer_count
                if wafer_count > 0:
                    self.processing_wafer_count += 1
                    if wafer_type:
                        item['wafer_type'] = wafer_type
                item['process_time_remaining'] = process_time_remaining
                item['residency_time_remaining'] = residency_time_remaining
                item['wafer_start_time'] = wafer_start_time
                item['processed_time'] = wafer_processed_time
                item['residency_time'] = wafer_residency_time

    # ======= 更新LL的状态 =======
    def update_loadlock_state(self, entry_count, exit_count, wafer_type=None):
        self.entry_wafer = entry_count  # 待加工的晶圆数量
        self.exit_wafer = exit_count  # 清除回到LL的晶圆数量
        if wafer_type:
            self.wafer_type_count[wafer_type] += 1

    # ======= 更新robot的状态 =======
    def update_robot_state(self, robot_current_module, robot_wait_time):
        if robot_current_module == 'LL':
            self.robot_current_module = 0
        else:
            self.robot_current_module = self.modules_name.index(robot_current_module) + 1
        self.robot_wait_time = robot_wait_time

    def update_system_wafer_state(self):
        self.processing_wafer_count = 0
        for item in self.state_values:
            if item['wafer_count'] > 0:
                self.processing_wafer_count += 1

    # ======= 获取状态 =======
    def get_state(self):
        self.state = list()
        self.state.append(self.exit_wafer)  # 加工完成后回到LL的晶圆数量
        self.state.append(self.entry_wafer)  # 待加工的晶圆数量
        self.state.append(self.robot_current_module)  # robot当前位置
        self.state.append(self.processing_wafer_count)  # 组合设备系统中当前正在加工的晶圆数量

        # 添加两种晶圆的计数
        self.state.append(self.wafer_type_count[1])
        self.state.append(self.wafer_type_count[2])

        for item in self.state_values:
            self.state.append(item['wafer_count'])
            self.state.append(0 if item['wafer_count'] == 0 else item['wafer_type'])  # 晶圆类型
            self.state.append(item['process_time_remaining'])
            self.state.append(item['residency_time_remaining'])
            self.state.append(item['processed_time'])
            self.state.append(item['residency_time'])

        return self.state

    def get_state_dim(self):
        return len(self.get_state())

    # ======= 获取奖励 =======
    def get_reward(self, robot_action_flag, fail_flag, bottleneck_time):
        success_flag = False
        self.reward = 0
        if fail_flag:
            self.reward = -2000
            success_flag = False

        if self.exit_wafer == self.total_wafer:  # 系统中晶圆已经清空完毕
            self.reward += 2000
            success_flag = True

        if self.robot.current_module == 'LL' and self.pre_exit_wafer + 1 == self.exit_wafer:  # 说明有新的晶圆回到LL
            self.reward += 15
            self.pre_exit_wafer = self.exit_wafer

        self.reward += self.processing_wafer_count * 10

        if self.robot.take_action_time > 0:
            self.reward -= self.robot.take_action_time

        # 增加中间奖励，鼓励探索不同动作
        if robot_action_flag:
            self.reward += 10

        return self.reward, success_flag

    # ======= 打印信息 =======
    def print_info(self, reward, env):
        print('------------------------------------------------执行动作完毕后的系统状态--------------')
        print('LL中尚未加工的晶圆数:   {0}'.format(self.entry_wafer))
        print('加工完成后返回LL的晶圆数:   {0}'.format(self.exit_wafer))
        print('当前系统中正在加工的晶圆数量:   {0}'.format(self.processing_wafer_count))
        print('本次动作robot等待时间:   {0}'.format(self.robot.wait_time))
        print('robot完成动作时间:   {0}'.format(self.robot.take_action_time))
        print(f"robot当前位于:{self.robot.current_module}")
        for item in self.state_values:
            print('{0}\t是否存在晶圆:{1}\t晶圆类型:{2}\t晶圆剩余加工时间:{3}\t剩余驻留时间:{4}\t晶圆已经加工时间:{5}\t晶圆已经驻留时间时间:{6}\t'
                  '晶圆在当前PM开始加工时间:{7}'.format(item['name'],
                                                        item['wafer_count'],
                                                        item['wafer_type'] if item['wafer_count'] > 0 else 'N/A',
                                                        item['process_time_remaining'],
                                                        item['residency_time_remaining'],
                                                        item['processed_time'],
                                                        item['residency_time'],
                                                        item['wafer_start_time']))

        print('当前奖励: {0}  当前时间:{1}'.format(reward, env.now))


# -------------------------------------------------
#   晶圆类（无修改）
# -------------------------------------------------
class Wafer(object):
    def __init__(self, id, state, is_virtual, process_time, max_stay_time, wait_time, wafer_type):
        self.id = id
        self.state = state  # 晶圆当前所在的加工模块
        self.is_broken = False  # 晶圆是否损坏
        self.is_virtual = is_virtual  # 晶圆是否为虚拟晶圆
        self.process_time = process_time  # 加工时间字典（模块名: 时间）
        self.max_stay_time = max_stay_time  # 最大驻留时间字典（模块名: 时间）
        self.wait_time = wait_time  # 等待时间
        self.wafer_type = wafer_type  # 晶圆类型(1或2)
        self.process_step = 0  # 当前处理步骤
        self.path = self.get_path()  # 晶圆的加工路径（支持并行模块）
        self.selected_parallel_module = None  # 记录并行模块中选择的具体模块

    def get_path(self):
        """根据晶圆类型返回对应的加工路径（不含并行模块）
        单臂单组合设备调度环境
        LL->PM1->PM2->PM3->PM4->LL
        LL->PM1->PM2->PM5->PM6->LL
        """
        if self.wafer_type == 1:
            # 类型1路径：共享路径+PM3+PM4+LL
            return ["LL", "PM1", "PM2", "PM3", "PM4", "LL"]
        elif self.wafer_type == 2:
            # 类型2路径：共享路径+PM5+PM6+LL
            return ["LL", "PM1", "PM2", "PM5", "PM6", "LL"]
        else:
            return []

    def get_possible_next_modules(self):
        """获取当前步骤的所有可能下一个模块（处理并行逻辑）"""
        if self.process_step >= len(self.path) - 1:
            return None  # 已完成所有步骤
        current_next = self.path[self.process_step + 1]
        if isinstance(current_next, list):
            # 并行模块，返回所有可能选项
            return current_next
        else:
            # 串行模块，返回单一选项
            return [current_next]

    def confirm_next_module(self, selected_module):
        """确认选择的并行模块，更新路径（仅并行步骤需要）"""
        if self.process_step + 1 >= len(self.path):
            return False
        current_next = self.path[self.process_step + 1]
        if isinstance(current_next, list) and selected_module in current_next:
            # 记录选择的并行模块，用于后续步骤校验
            self.selected_parallel_module = selected_module
            return True
        return False


# -------------------------------------------------
#   PM类（无修改）
# -------------------------------------------------
class ProcessModule(object):
    def __init__(self, env, name, process_time, max_stay_time, wait_time, pre_module, next_module, loadlock, robot,
                 module_list, wafer_types_allowed=None):
        self.env = env
        self.name = name
        self.process_time = process_time
        self.max_stay_time = max_stay_time
        self.wait_time = wait_time
        self.pre_module = pre_module
        self.next_module = next_module
        self.loadlock = loadlock
        self.robot = robot
        self.module_list = module_list
        self.state = None
        self.store = simpy.Store(self.env)
        self.monitoring_data = []
        self.wafer_start_time = 0  # 当前晶圆进入PM的时间
        self.last_wafer_left_time = 0  # 上一个晶圆离开的时间
        self.idle_time = 0
        self.fail = False  # 失败标记
        self.wafer_types_allowed = wafer_types_allowed or []  # 允许处理的晶圆类型

    # ======= 加载晶圆到PM（核心修改：支持并行路径校验） =======
    def load(self, wafer):
        if self.store.items.__len__() == 1:
            print(f"{self.name}加载晶圆失败,已有晶圆存在")
            self.fail = True
            return
        # 检查晶圆类型是否允许在此PM处理
        if self.wafer_types_allowed and wafer.wafer_type not in self.wafer_types_allowed:
            print(f"{self.name}不允许处理类型{wafer.wafer_type}的晶圆")
            self.fail = True
            return
        # 检查晶圆来源是否正确（支持并行模块逻辑）
        possible_next_modules = wafer.get_possible_next_modules()
        if possible_next_modules is None:
            print(f"{self.name}加载晶圆失败，晶圆已完成所有步骤")
            self.fail = True
            return
        # 校验当前模块是否在可能的下一个模块中（含并行选项）
        if self.name not in possible_next_modules:
            print(f"{self.name}加载晶圆失败，加工顺序不匹配。允许的下一个模块: {possible_next_modules}，实际: {self.name}")
            self.fail = True
            return
        # 若当前步骤是并行模块，确认晶圆已选择该模块
        if isinstance(wafer.path[wafer.process_step + 1], list):
            if not wafer.confirm_next_module(self.name):
                print(f"{self.name}加载晶圆失败，未确认并行模块选择")
                self.fail = True
                return

        self.fail = False
        self.max_stay_time = wafer.max_stay_time[self.name]
        self.process_time = wafer.process_time[self.name]
        self.store.put(wafer)
        self.wafer_start_time = self.env.now  # 记录晶圆开始进入PM的时间
        self.env.process(self.processing(self.process_time))  # 启动加工进程

    def processing(self, process_time):
        yield self.env.timeout(process_time)  # 模拟加工晶圆
        self.store.items[0].state = self.name  # 加工完成，标记晶圆状态为当前模块名称
        # 更新晶圆处理步骤（并行模块仅算1步）
        self.store.items[0].process_step += 1

    def unload(self):
        if self.store.items.__len__() == 0:
            self.fail = True
            print(f"{self.name}卸载晶圆失败，当前PM不存在晶圆")
            return None

        if self.env.now - self.wafer_start_time < self.process_time and self.state != 'fail':
            self.fail = True
            print(f"{self.name}卸载晶圆失败，未到达加工时间")
            return None

        if self.env.now - self.wafer_start_time > self.process_time + self.max_stay_time and self.state != 'fail':
            self.fail = True
            print(f"{self.name}卸载晶圆失败，违反了驻留时间约束")
            wafer = self.store.get()
            wafer.is_broken = True  # 晶圆损坏
            if type(wafer) is simpy.resources.store.StoreGet:
                return wafer.value
            return wafer

        self.fail = False

        wafer = self.store.get()
        self.last_wafer_left_time = self.env.now
        self.wafer_start_time = 0
        if type(wafer) is simpy.resources.store.StoreGet:
            return wafer.value
        return wafer

    # ======= 获取当前PM的晶圆数量 =======
    def get_wafer_count(self):
        return self.store.items.__len__()

    # ======= 获取当前PM的晶圆 =======
    def get_current_wafer(self):
        if self.store.items.__len__() == 1:
            return self.store.items[-1]
        return None

    # ======= 获取当前PM的晶圆的状态 =======
    def get_wafer_state(self):
        if self.store.items.__len__() == 1:
            return self.store.items[-1].is_broken
        return None

    # ======= 获取当前PM的晶圆剩余的加工时间 =======
    def get_process_remaining_time(self):
        if self.store.items.__len__() != 0:  # 当前存在晶圆
            wafer = self.store.items[0]
            self.process_time = wafer.process_time[self.name]
            return self.process_time - self.env.now + self.wafer_start_time
        return 0

    # ======= 获取当前PM的晶圆剩余的驻留时间 =======
    def get_residency_remaining_time(self):
        if self.store.items.__len__() != 0:  # 当前存在晶圆
            wafer = self.store.items[0]
            self.max_stay_time = wafer.max_stay_time[self.name]
            self.process_time = wafer.process_time[self.name]
            upper_limit = self.max_stay_time + self.process_time
            residence_time = self.env.now - self.wafer_start_time
            return (upper_limit - residence_time)
        return 0  # 神经网络不能输入无穷值，返回0,则说明不存在晶圆

    # ======= 获取当前PM的两次加工之间的空闲时间 =======
    def get_idle_time(self):
        return 0

    # ======= 获取robot在PM前等待卸载晶圆的时间 =======
    def get_wait_time(self):
        current_time = self.env.now
        if self.store.items.__len__() == 0:  # 当前PM没有晶圆
            self.fail = True
            return 0
        else:
            wafer = self.store.items[0]
            self.process_time = wafer.process_time[self.name]
            self.wafer_maximum_residency = wafer.max_stay_time[self.name]
            wait_time = self.wafer_start_time + self.process_time - current_time
            if wait_time >= 0:  # 尚未加工完成，需要在当前PM前等待
                self.fail = False
                return wait_time
            elif wait_time < 0 and wait_time + self.max_stay_time >= 0:  # 加工完成，但没有违反晶圆驻留时间约束
                self.fail = False
                return 0
            else:  # 违反驻留时间约束
                self.fail = True
                return wait_time + self.wafer_maximum_residency

    # ======= 获取当前PM的晶圆已经加工时间 =======
    def get_wafer_processed_time(self):
        if self.store.items.__len__() == 0:  # 当前不存在晶圆
            return -1
        if self.env.now - self.wafer_start_time > self.process_time:  # 如果已经加工完成，则返回模块的加工时间
            return self.process_time
        return self.env.now - self.wafer_start_time  # 返回晶圆实际加工时间

    def get_wafer_residency_time(self):
        if self.store.items.__len__() == 0:  # 当前不存在晶圆
            return -1
        return self.env.now - self.wafer_start_time


# -------------------------------------------------
#   LL类（无修改）
# -------------------------------------------------
class Loadlock(object):
    def __init__(self, env, name, wafer_num, module_list):
        self.env = env
        self.name = name
        self.wafer_num = wafer_num  # 系统中的晶圆数量
        # 按类型分存储队列（类型1和类型2）
        self.entry_stores = {1: simpy.Store(self.env), 2: simpy.Store(self.env)}  # 按类型存储待加工晶圆
        self.exit_store = simpy.Store(self.env)  # 存储已完成晶圆
        self.monitoring_data = []
        self.system_state = "initial_transient"
        self.fail = False
        self.virtual_wafer_num = 0
        self.entry_system_wafer_num = 0
        self.module_list = module_list
        self.last_unload_time = 0
        self.idle_time = 0
        self.system_wafer_count = 0
        self.wafer_start_time = 0
        self.wafer_type_count = {1: 0, 2: 0}  # 记录每种类型晶圆的数量

    def initialize(self, wafers):
        # 按类型放入对应队列
        # 初始化时强制清零，避免残留值影响
        self.wafer_type_count = {1: 0, 2: 0}
        for wafer in wafers:
            self.entry_stores[wafer.wafer_type].put(wafer)
            self.wafer_type_count[wafer.wafer_type] += 1  # 初始化时正确计数
        print(f"LL初始化完成: 类型1={self.wafer_type_count[1]}, 类型2={self.wafer_type_count[2]}")
    # ======= 加载晶圆到LL =======
    def load(self, wafer):
        # 核心修改：更新完成校验逻辑，两种类型晶圆步骤数不同
        if wafer.wafer_type == 1 and wafer.process_step == 4:  # 类型1晶圆完成4个步骤（PM5之后）
            pass  # 允许返回LL
        elif wafer.wafer_type == 2 and wafer.process_step == 4:  # 类型2晶圆完成4个步骤（PM6之后）
            pass  # 允许返回LL
        else:
            self.fail = True
            print(f"W{wafer.id}加载到LL失败，晶圆尚未完成！当前步骤: {wafer.process_step}，类型: {wafer.wafer_type}")
            return

        self.fail = False
        self.exit_store.put(wafer)
        self.wafer_type_count[wafer.wafer_type] -= 1
        if self.exit_store.items.__len__() == self.wafer_num:  # 已经清空完毕
            print(f"\n==========================加工完毕:{self.env.now}===========================")
            print(f"清空用时:{self.env.now}")

    # ======= 从LL卸载晶圆 =======
    def unload(self, wafer_type=None):
        # 情况1：未指定类型，返回存在的类型
        if wafer_type is None:
            if self.wafer_type_count[1] > 0:
                wafer_type = 1
            elif self.wafer_type_count[2] > 0:
                wafer_type = 2
            else:
                self.fail = True
                print("卸载失败：无任何待加工晶圆")
                return None

        # 情况2：指定类型存在，正常返回
        if self.entry_stores[wafer_type].items:
                self.fail = False
                # 先获取实际晶圆（确保取到的是目标类型）
                wafer = self.entry_stores[wafer_type].get()
                if isinstance(wafer, simpy.resources.store.StoreGet):
                    wafer = wafer.value
                # 验证晶圆类型（关键：防止取错类型导致计数错误）
                if wafer.wafer_type != wafer_type:
                    print(f"错误：取出的晶圆类型{wafer.wafer_type}与请求类型{wafer_type}不符！")
                    self.entry_stores[wafer.wafer_type].put(wafer)  # 放回错误取出的晶圆
                    self.fail = True
                    return None
                # 正确更新计数
                self.wafer_type_count[wafer_type] -= 1
                self.last_unload_time = self.env.now + 10
                self.system_wafer_count += 1
                return wafer
        # 情况3：指定类型不存在，但有其他类型（容错处理）
        other_type = 3 - wafer_type
        if self.entry_stores[other_type].items:
            self.fail = False  # 不标记失败
            print(f"警告：类型{wafer_type}已耗尽，自动切换到类型{other_type}")
            wafer = self.entry_stores[other_type].get()
            self.last_unload_time = self.env.now + 10
            self.system_wafer_count += 1
            self.wafer_type_count[other_type] -= 1
            return wafer.value if isinstance(wafer, simpy.resources.store.StoreGet) else wafer

        # 情况4：无任何晶圆，才标记失败
        self.fail = True
        print(f"卸载失败：所有类型晶圆均已耗尽")
        return None

    # ======= 剩余加工晶圆数量=======
    def get_remaining_wafer_count(self):
        return sum(store.items.__len__() for store in self.entry_stores.values())

    # ======= 已完成加工晶圆数量=======
    def get_finished_wafer_count(self):
        return self.exit_store.items.__len__()


# -------------------------------------------------
#   Robot类（无修改）
# -------------------------------------------------
class Robot(object):
    def __init__(self, env, name, move_time, work_time, unload_time_LL, current_module, loadlock):
        self.env = env
        self.name = name
        self.store = simpy.Store(self.env)
        self.unload_time_LL = unload_time_LL
        self.move_time = move_time
        self.work_time = work_time
        self.monitoring_data = []
        self.current_module = current_module
        self.pre_module = None
        self.wafer_start_time = 0
        self.wait_time = 0
        self.loadlock = loadlock
        self.take_action_time = 0
        self.action_start_time = 0
        self.fail = False
        self.carrying_wafer_type = None  # 记录当前携带的晶圆类型

    # ======= 加载晶圆到robot =======
    def load(self, wafer):
        if self.store.items.__len__() == 1:
            self.fail = True
            return
        self.fail = False
        if self.current_module == "LL":
            yield self.env.timeout(self.unload_time_LL)
        else:
            yield self.env.timeout(self.work_time)
        self.store.put(wafer)
        self.wafer_start_time = self.env.now
        self.carrying_wafer_type = wafer.wafer_type  # 记录晶圆类型

    # ======= 从robot卸载晶圆 =======
    def unload(self):
        if self.store.items.__len__() == 0:
            self.fail = True
            return None
        self.fail = False
        wafer = self.store.get()
        self.wafer_start_time = 0
        self.carrying_wafer_type = None  # 清空携带的晶圆类型
        yield self.env.timeout(self.work_time)
        if type(wafer) is simpy.resources.store.StoreGet:
            return wafer.value
        return wafer

    # ======= 获取当前robot的晶圆数量 =======
    def get_wafer_count(self):
        return self.store.items.__len__()

    # ======= robot移动到目标模块 =======
    def move(self, target):
        if self.current_module != target:
            yield self.env.timeout(self.move_time)
            self.pre_module = self.current_module
            self.current_module = target


# -------------------------------------------------
#   环境类（核心修改：首次取片强制类型1）
# -------------------------------------------------
class Environment(object):

    def __init__(self, args, wafer_num, wafer_type_distribution=None):
        self.env = simpy.Environment()
        self.robot_actions = args.robot_actions  # 分别表示不持有晶圆移动，等待，卸载，持有晶圆移动，加载
        self.modules_list = args.modules_list
        self.steps_list = args.steps_list
        self.process_residency_time_dict = args.process_residency_time_dict
        self.process_residency_time_list = args.process_residency_time_list
        self.max_stay_time_dict = args.max_stay_time_dict
        self.max_stay_time_list = args.max_stay_time_list
        self.process_time_dict = args.process_time_dict
        self.process_time_list = args.process_time_list
        self.wait_time_dict = args.wait_time_dict
        self.wait_time_list = args.wait_time_list
        self.unload_time_LL = args.unload_time_LL
        self.work_time = args.work_time
        self.move_time = args.move_time
        self.wafer_num = wafer_num
        self.wafer_type_distribution = wafer_type_distribution or [0.5, 0.5]  # 默认两种晶圆各占50%

        # 核心修改1：新增首次取片标记（初始为True，取片后改为False）
        self.first_wafer_taken = False  
        # 核心修改2：记录下一次应该选择的类型，确保严格交替
        self.next_target_type = 1  # 初始下一次应选类型为1
        # 添加调试变量，记录交替序列
        self.sequence_log = []

        self.robot = None
        self.loadlock = None
        self.modules = list()
        self.bottleneck_time = 0
        self.state = list()
        self.reward = 0
        self.done = False
        self.fail_flag = False
        self.success_flag = False
        self.state_dim = 0
        self.action_dim = args.action_dim
        self.actions = args.actions
        self.robot_current_module = 'LL'  # 调度开始
        self.initialize()

    def initialize(self):
        self.env = simpy.Environment()  # 创建环境

        # 生成晶圆
        wafers = self.generate_wafers(self.wafer_num, self.wafer_type_distribution)  # 编号从1开始

        # 生成loadlock对象
        self.loadlock = Loadlock(self.env, "LL", self.wafer_num, self.modules_list)
        self.loadlock.initialize(wafers)

        # 核心修改：调整PM允许处理的晶圆类型
        # 共享路径模块(PM1-PM2)允许处理两种类型，PM3-PM4仅处理类型1，PM5-PM6仅处理类型2
        pm_wafer_types = {
            "PM1": [1, 2],  # 两种类型都可以处理
            "PM2": [1, 2],  # 两种类型都可以处理
            "PM3": [1, ],  # 仅处理类型1
            "PM4": [1, ],  # 仅处理类型1
            "PM5": [2, ],  # 仅处理类型2
            "PM6": [2, ]  # 仅处理类型2
        }

        # 生成robot对象
        self.robot = Robot(self.env, "robot", self.move_time, self.work_time, self.unload_time_LL,
                           self.robot_current_module, self.loadlock)

        # 初始化所有PM
        self.modules.clear()
        for i in range(1, len(self.modules_list)):  # 所有PM
            pm_name = self.steps_list[i]
            pre_pm = self.steps_list[i - 1]
            next_pm = self.steps_list[i + 1] if i < len(self.steps_list) - 1 else None
            PM = ProcessModule(self.env, pm_name, self.process_time_dict[pm_name], self.max_stay_time_dict[pm_name],
                               self.wait_time_dict[pm_name],
                               pre_pm, next_pm, self.loadlock, self.robot, self.modules,
                               wafer_types_allowed=pm_wafer_types.get(pm_name, []))

            self.modules.append(PM)

        # 初始化各事件
        self.event_entry = self.env.event()
        self.event_exit = self.env.event()
        self.event_hdlr = self.env.event()
        self.event_step = self.env.event()
        self.event_action = self.env.event()
        self.events_robot_action = {event: self.env.event() for event in self.robot_actions}
        if not self.events_robot_action["IDLE"].triggered:
            self.events_robot_action["IDLE"].succeed()  # 初始时，robot处于IDLE状态

        # 初始化动作、奖励、状态和各种标记
        self.action, self.reward = 0, 0
        self.fail_flag, self.success_flag, self.done = False, False, False
        self.robot_action_flag = False
        self.state = []
        self.wafer_in_proc = 0

        self.curr_nope_count = 0
        self.profiler = self.init_system_profiler()
        self.state_dim = self.profiler.get_state_dim()
        self.bottleneck_time = sum(self.process_time_list)
        self.process_handler = self.env.process(self.proc_handler())  # 处理器进程

    # ======= 环境的外部接口：重置环境（无修改） =======
    def reset(self):
        del self.env
        # 重置时恢复首次取片标记为True
        self.first_wafer_taken = False
        self.initialize()
        # 更新状态
        for pm in self.modules:
            wafer = pm.get_current_wafer()
            wafer_type = wafer.wafer_type if wafer else None
            self.profiler.update_modules_state(pm.name,
                                               pm.store.items.__len__(),
                                               pm.get_process_remaining_time(),
                                               pm.get_residency_remaining_time(),
                                               pm.state,
                                               pm.wafer_start_time,
                                               pm.get_wafer_state(),
                                               pm.get_wafer_processed_time(),
                                               pm.get_wafer_residency_time(),
                                               wafer_type)

        self.profiler.update_loadlock_state(self.loadlock.get_remaining_wafer_count(),
                                            self.loadlock.get_finished_wafer_count())
        self.profiler.update_robot_state(self.robot.current_module, self.robot.wait_time)
        self.profiler.update_system_wafer_state()
        self.state = self.profiler.get_state()
        return self.state

    # ======= 环境的外部接口：执行动作（无修改） =======
    def step(self, action):
        self.fail_flag = False
        self.action = action
        self.event_action.succeed()  # 触发event_action，即通知处理器proc_handler需要处理动作
        print("******开始时间：", self.env.now)
        self.robot.action_start_time = self.env.now
        self.env.run(self.event_step)  # 运行环境，直到event_step触发
        self.event_step = self.env.event()
        print("******结束时间：", self.env.now)
        self.f_time = self.env.now
        obs = self.get_observation()

        if self.done:
            return obs
        return obs

    # ======= 初始化分析器（无修改） =======
    def init_system_profiler(self):
        pm_names = list()
        for pm in self.modules:
            pm_names.append(pm.name)
        profiler = system_profiler(pm_names, self.modules, self.loadlock, self.robot, self.wafer_num)
        return profiler

    # ======= 获取观测值obs（无修改） =======
    def get_observation(self):
        for pm in self.modules:
            if pm.store.items.__len__() == 1 and pm.get_residency_remaining_time() < 0:
                self.fail_flag = True
                print(f"{pm.name}违反驻留时间约束")
                break
        # 更新状态
        for pm in self.modules:
            wafer = pm.get_current_wafer()
            wafer_type = wafer.wafer_type if wafer else None
            self.profiler.update_modules_state(pm.name,
                                               pm.store.items.__len__(),
                                               pm.get_process_remaining_time(),
                                               pm.get_residency_remaining_time(),
                                               pm.state,
                                               pm.wafer_start_time,
                                               pm.get_wafer_state(),
                                               pm.get_wafer_processed_time(),
                                               pm.get_wafer_residency_time(),
                                               wafer_type)

        self.profiler.update_loadlock_state(self.loadlock.get_remaining_wafer_count(),
                                            self.loadlock.get_finished_wafer_count())
        self.profiler.update_robot_state(self.robot.current_module, self.robot.wait_time)

        self.profiler.update_system_wafer_state()
        # 获取状态
        self.state = self.profiler.get_state()

        # 获取奖励
        self.reward, self.success_flag = self.profiler.get_reward(self.robot_action_flag, self.fail_flag,
                                                                  self.bottleneck_time)
        # 打印信息
        self.profiler.print_info(self.reward, self.env)

        if self.robot_action_flag and not self.fail_flag:
            print("---------成功执行动作--------")

        if self.fail_flag is True:  # 失败标记为真，结束
            self.done = True
            print(
                "**********************************************************失败Terminate state!!!**********************************************************")
        elif self.success_flag:
            self.done = True
            print(
                "**********************************************************成功Terminate state!!!**********************************************************")

        else:
            self.done = False

        return self.state, self.reward, self.done

    # ======= 处理器（无修改） =======
    def proc_handler(self):
        while True:
            yield (self.event_action)
            if self.event_action.triggered:
                self.event_action = self.env.event()
            timeout_no_op = 1
            self.robot_action_flag = False
            action_taken = int(self.action)
            self.robot.action_start_time = self.env.now
            if action_taken >= 0 and action_taken < len(self.actions):
                self.curr_nope_count = 0
                self.env.process(self.execute_action(self.actions[action_taken]))
            else:
                self.fail_flag = True
                if not self.event_step.triggered:
                    self.event_step.succeed()
                return

    # ======= robot执行动作（核心修改：首次取片强制类型1） =======
    def execute_action(self, action):
        print(f"\n===========执行动作{action}===========")
        source_idx, target_idx = action
        source_module = self.modules_list[source_idx]
        target_module = self.modules_list[target_idx]
        target_wafer_type = None

        # 仅处理LL→PM1的动作（[0,1]）
        if source_module == "LL" and target_module == "PM1":
            # 打印当前状态（关键调试信息）
            print(f"当前LL库存：类型1={self.loadlock.wafer_type_count[1]}, 类型2={self.loadlock.wafer_type_count[2]}")
            print(f"首次取片标记：{self.first_wafer_taken}（True=已取过，False=未取过）")

            # 核心修改：首次取片强制类型1
            if not self.first_wafer_taken:
                required_type = 1  # 首次取片强制类型1
                print(f"【首次取片】强制选择类型1，忽略默认交替逻辑")
            else:
                required_type = self.next_target_type  # 非首次取片，按原交替逻辑

            # 2. 实际库存检查（直接检查entry_stores的实际数量，而非依赖wafer_type_count）
            actual_count = len(self.loadlock.entry_stores[required_type].items)
            has_required = actual_count > 0

            # 3. 强制修正：如果wafer_type_count与实际库存不符，同步计数
            if self.loadlock.wafer_type_count[required_type] != actual_count:
                print(
                    f"警告：类型{required_type}计数不一致（记录={self.loadlock.wafer_type_count[required_type]}, 实际={actual_count}），已修正")
                self.loadlock.wafer_type_count[required_type] = actual_count

            # 4. 确定实际要取的类型（首次取片若类型1耗尽，才允许切换到类型2，避免仿真失败）
            if has_required:
                actual_type = required_type
            else:
                actual_type = 3 - required_type
                print(f"警告：类型{required_type}实际已耗尽，临时使用类型{actual_type}")
                # 首次取片若类型1耗尽，取类型2后仍标记为“已取过首次”
                if not self.first_wafer_taken:
                    print(f"【首次取片】类型1已耗尽，被迫取类型2，后续按交替逻辑执行")

            # 5. 记录序列、更新首次取片标记和下次目标
            self.sequence_log.append(actual_type)
            self.first_wafer_taken = True  # 无论取什么类型，首次取片标记都改为False
            self.next_target_type = 3 - actual_type  # 下次必为相反类型
            target_wafer_type = actual_type
            print(f"实际取出类型：{actual_type}，下次预期类型：{self.next_target_type}")
        
        # 其他动作的类型约束（无修改）
        elif target_module in ["PM2", "PM3", "PM4"]:
            target_wafer_type = None
        elif target_module == "PM5":
            target_wafer_type = 1
        elif target_module == "PM6":
            target_wafer_type = 2

        if not self.events_robot_action["MT"].triggered and self.events_robot_action['IDLE'].triggered:
            self.events_robot_action["MT"].succeed()

        # 1. 不携带晶圆移动到源模块（如果源是LL）
        if self.events_robot_action["MT"].triggered:
            print(f"时间:{self.env.now}\tRobot执行MT\t", self.robot.current_module, "——>", source_module)
            yield self.env.process(self.robot.move(source_module))

            if self.robot.current_module == "LL":
                current_pm = self.loadlock
            else:
                current_pm = self.modules[self.modules_list.index(self.robot.current_module) - 1]

            self.events_robot_action["MT"] = self.env.event()
            if not self.events_robot_action["WT"].triggered:
                self.events_robot_action["WT"].succeed()

        # 2. 等待
        if self.events_robot_action["WT"].triggered:
            self.robot.wait_time = wait_time = 0
            if current_pm.name != "LL":
                wait_time = current_pm.get_wait_time()
            print(f"时间:{self.env.now}\tRobot执行等待WT\t当前模块:{self.robot.current_module}\t等待时间为:{wait_time}")
            if wait_time >= 0:
                yield self.env.timeout(wait_time)
                self.robot.wait_time = wait_time
            self.fail_flag = current_pm.fail
            self.events_robot_action["WT"] = self.env.event()
            if self.fail_flag and not self.event_step.triggered:
                self.robot.take_action_time = self.env.now - self.robot.action_start_time
                self.event_step.succeed()
                return
            if not self.events_robot_action["UT"].triggered:
                self.events_robot_action["UT"].succeed()

        # 3. 卸载晶圆（从LL卸载时指定目标类型）
        if self.events_robot_action["UT"].triggered:
            print(f"时间:{self.env.now}\tRobot执行卸载UT\t当前模块:{current_pm.name}")
            # 如果是从LL卸载，使用目标模块对应的类型取晶圆
            if current_pm.name == "LL" and target_wafer_type is not None:
                wafer_ut = current_pm.unload(wafer_type=target_wafer_type)  # 按类型取出
            else:
                wafer_ut = current_pm.unload()  # 非LL模块按原逻辑

            # 检查卸载结果
            if wafer_ut is None:
                self.fail_flag = True
                print(f"卸载失败：{current_pm.name}返回空晶圆，终止动作")
                if not self.event_step.triggered:
                    self.event_step.succeed()
                return

            # 加载到Robot
            yield self.env.process(self.robot.load(wafer_ut))
            self.events_robot_action["UT"] = self.env.event()

            if (current_pm.fail or self.robot.fail) and not self.event_step.triggered:
                self.fail_flag = True
                self.robot.take_action_time = self.env.now - self.robot.action_start_time
                self.event_step.succeed()
                return

            if not self.events_robot_action["CT"].triggered:
                self.events_robot_action["CT"].succeed()

        # 4. 携带晶圆移动到目标模块
        if self.events_robot_action["CT"].triggered:
            print(
                f"时间:{self.env.now}\tRobot执行CT\t{self.robot.current_module}——>{target_module}\t当前晶圆为:W{wafer_ut.id}")
            yield self.env.process(self.robot.move(target_module))
            if self.robot.current_module == "LL":
                current_pm = self.loadlock
            else:
                current_pm = self.modules[self.modules_list.index(self.robot.current_module) - 1]
            self.events_robot_action["CT"] = self.env.event()

            if not self.events_robot_action["LT"].triggered:
                self.events_robot_action["LT"].succeed()

        # 5. 加载到目标模块
        if self.events_robot_action["LT"].triggered:
            print(f"时间:{self.env.now}\tRobot执行加载LT\t当前模块:{self.robot.current_module}")
            wafer_lt = yield self.env.process(self.robot.unload())
            current_pm.load(wafer_lt)

            if (current_pm.fail or self.robot.fail):
                self.fail_flag = True
            else:
                self.fail_flag = False

            if not self.event_step.triggered:
                self.event_step.succeed()
            print("完成动作时间:", self.env.now)
            self.robot.take_action_time = self.env.now - self.robot.action_start_time

            if not self.fail_flag:
                self.profiler.reward += 10  # 增加中间奖励,将某PM的晶圆成功移动到下一PM
                self.robot_action_flag = True
                self.events_robot_action["LT"] = self.env.event()
                if not self.events_robot_action["IDLE"].triggered:
                    self.events_robot_action["IDLE"].succeed()

    # ======= 生成晶圆（无修改） =======
    def generate_wafers(self, wafer_num, distribution):
        """生成指定数量和类型分布的晶圆"""
        wafer_list = list()
        type1_count = int(wafer_num * distribution[0])
        for i in range(wafer_num):
            wafer_type = 1 if i < type1_count else 2
            wafer_list.append(Wafer(i + 1, 'LL', False, self.process_time_dict, self.max_stay_time_dict,
                                    self.wait_time_dict, wafer_type))
        print(f"生成{wafer_num}个晶圆: 类型1={type1_count}, 类型2={wafer_num - type1_count}")
        return wafer_list

    # ======= 生成动作掩码（无修改） =======
    def get_mask(self):
        """生成动作掩码：合法动作=1，非法动作=0"""
        mask = [True] * len(self.actions)  # 初始化所有动作都允许

        # 获取各PM的晶圆数量（索引对应modules_list中的PM1-PM6）
        pm1_wafer_count = self.modules[0].get_wafer_count() if len(self.modules) > 0 else 0  # PM1
        pm2_wafer_count = self.modules[1].get_wafer_count() if len(self.modules) > 1 else 0  # PM2
        pm3_wafer_count = self.modules[2].get_wafer_count() if len(self.modules) > 2 else 0  # PM3
        pm4_wafer_count = self.modules[3].get_wafer_count() if len(self.modules) > 3 else 0  # PM4
        pm5_wafer_count = self.modules[4].get_wafer_count() if len(self.modules) > 4 else 0  # PM5
        pm6_wafer_count = self.modules[5].get_wafer_count() if len(self.modules) > 5 else 0  # PM6

        # 其他PM（除目标PM外的所有PM）的晶圆数量总和
        other_pms_when_pm1 = pm2_wafer_count + pm3_wafer_count + pm4_wafer_count + pm5_wafer_count + pm6_wafer_count
        other_pms_when_pm2 = pm1_wafer_count + pm3_wafer_count + pm4_wafer_count + pm5_wafer_count + pm6_wafer_count
        other_pms_when_pm3 = pm1_wafer_count + pm2_wafer_count + pm4_wafer_count + pm5_wafer_count + pm6_wafer_count
        other_pms_when_pm4 = pm1_wafer_count + pm2_wafer_count + pm3_wafer_count + pm5_wafer_count + pm6_wafer_count
        other_pms_when_pm5 = pm1_wafer_count + pm2_wafer_count + pm3_wafer_count + pm4_wafer_count + pm6_wafer_count
        other_pms_when_pm6 = pm1_wafer_count + pm2_wafer_count + pm3_wafer_count + pm4_wafer_count + pm5_wafer_count
        all_pms = pm1_wafer_count + pm2_wafer_count + pm3_wafer_count + pm4_wafer_count + pm5_wafer_count + pm6_wafer_count

        #  所有PM不存在晶圆时，只允许[0,1]
        if all_pms == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1]]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        #  PM1存在晶圆，其他PM不存在时，只允许[1,2]
        elif pm1_wafer_count > 0 and other_pms_when_pm1 == 0:
            for i, act in enumerate(self.actions):
                if act in [[1, 2]]:
                    mask[i] = True
                elif act in [[0, 1], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        #  PM2
        elif pm2_wafer_count > 0 and other_pms_when_pm2 == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1]]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        #  PM1和PM2同时存在，其他PM不存在时
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and (all_pms - pm1_wafer_count - pm2_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[2, 3], [2, 5]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [3, 4], [4, 0], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM1 PM3
        elif pm1_wafer_count > 0 and pm3_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm3_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[1, 2],[3, 4],]:
                    mask[i] = True
                elif act in [[0, 1], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM2 PM3
        elif pm2_wafer_count > 0 and pm3_wafer_count > 0 and (
                all_pms - pm2_wafer_count - pm3_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1], [3, 4]]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM2 PM4
        elif pm2_wafer_count > 0 and pm4_wafer_count > 0 and (
                    all_pms - pm2_wafer_count - pm4_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1],[2, 5],[4, 0],]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM4 PM5
        elif pm4_wafer_count > 0 and pm5_wafer_count > 0 and (
                    all_pms - pm4_wafer_count - pm5_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1],[4, 0],[5, 6],]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM5
        elif pm5_wafer_count > 0 and other_pms_when_pm5 == 0:
            for i, act in enumerate(self.actions):
                if act in [[5, 6]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM6
        elif pm6_wafer_count > 0 and other_pms_when_pm6 == 0:
            for i, act in enumerate(self.actions):
                if act in [[6, 0]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6]]:
                    mask[i] = False

        # PM1 PM2 PM3
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm3_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm3_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[2, 3],[2, 5],[3, 4]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM1 PM2 PM4
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm4_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm4_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[2, 3],[2, 5],]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [3, 4], [4, 0], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM1 PM4 PM5
        elif pm1_wafer_count > 0 and pm4_wafer_count > 0 and pm5_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm4_wafer_count - pm5_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[1, 2],[4, 0],[5, 6],]:
                    mask[i] = True
                elif act in [[0, 1], [2, 3], [3, 4], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM2 PM4 PM5
        elif pm2_wafer_count > 0 and pm4_wafer_count > 0 and pm5_wafer_count > 0 and (
                all_pms - pm2_wafer_count - pm4_wafer_count - pm5_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1],[2, 3],[5, 6],]:
                    mask[i] = True
                elif act in [[1, 2], [3, 4], [4, 0], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM1 PM2 PM4 PM5
        elif pm1_wafer_count and pm2_wafer_count > 0 and pm4_wafer_count > 0 and pm5_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm4_wafer_count - pm5_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[4, 0],[5, 6],]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM1 PM2 PM5
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm5_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm5_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[2, 3],[5, 6]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [3, 4], [4, 0], [2, 5], [6, 0]]:
                    mask[i] = False

        # PM1 PM2 PM6
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm6_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm6_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[2, 3],[2, 5],[6, 0]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [3, 4], [4, 0], [5, 6]]:
                    mask[i] = False

        # PM1 PM3 PM6
        elif pm1_wafer_count > 0 and pm3_wafer_count > 0 and pm6_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm3_wafer_count - pm6_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[1, 2],[3, 4],[6, 0]]:
                    mask[i] = True
                elif act in [[0, 1], [2, 3], [4, 0], [2, 5], [5, 6],]:
                    mask[i] = False

        # PM2 PM3 PM6
        elif pm2_wafer_count > 0 and pm3_wafer_count > 0 and pm6_wafer_count > 0 and (
                all_pms - pm2_wafer_count - pm3_wafer_count - pm6_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[0, 1], [6, 0]]:
                    mask[i] = True
                elif act in [[1, 2], [2, 3], [3, 4], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        # PM1 PM2 PM3 PM6
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm3_wafer_count > 0 and pm6_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm3_wafer_count - pm6_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[3, 4],[6, 0]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [2, 3], [4, 0], [2, 5], [5, 6]]:
                    mask[i] = False

        # PM1 PM2 PM3
        elif pm1_wafer_count > 0 and pm2_wafer_count > 0 and pm3_wafer_count > 0 and (
                all_pms - pm1_wafer_count - pm2_wafer_count - pm3_wafer_count) == 0:
            for i, act in enumerate(self.actions):
                if act in [[3, 4]]:
                    mask[i] = True
                elif act in [[0, 1], [1, 2], [2, 3], [4, 0], [2, 5], [5, 6], [6, 0]]:
                    mask[i] = False

        return mask



if __name__ == "__main__":

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_actions', default=robot_actions, type=list)
    parser.add_argument('--modules_list', default=modules_list, type=list)
    parser.add_argument('--steps_list', default=steps_list, type=list)
    parser.add_argument('--process_residency_time_dict', default=process_residency_time_dict, type=dict)
    parser.add_argument('--process_residency_time_list', default=process_residency_time_list, type=list)
    parser.add_argument('--max_stay_time_dict', default=max_stay_time_dict, type=dict)
    parser.add_argument('--max_stay_time_list', default=max_stay_time_list, type=list)
    parser.add_argument('--process_time_dict', default=process_time_dict, type=dict)
    parser.add_argument('--process_time_list', default=process_time_list, type=list)
    parser.add_argument('--wait_time_dict', default=wait_time_dict, type=dict)
    parser.add_argument('--wait_time_list', default=wait_time_list, type=list)
    parser.add_argument('--unload_time_LL', default=unload_time_LL, type=int)
    parser.add_argument('--work_time', default=work_time, type=int)
    parser.add_argument('--move_time', default=move_time, type=int)
    parser.add_argument('--actions', default=actions, type=list)
    parser.add_argument('--action_dim', default=action_dim, type=int)
    parser.add_argument('--wafer_num', default=wafer_num, type=int)
    args = parser.parse_args()

    env = Environment(args, wafer_num, wafer_type_distribution)
    env.get_observation()

    # 适用于10个晶圆的测试动作序列（首次动作必为[0,1]，确保首次取片）
    test_actions = [0, 1, 0, 2, 1, 0, 3, 5, 1, 0, 4, 2, 6, 1, 0, 3, 5, 7, 1, 0, 4, 6, 2, 1, 0, 7, 3, 5, 1, 0, 4, 6, 2, 1, 0, 7, 3, 5, 1, 0, 4, 6, 2, 1, 7, 3, 5, 4, 6, 7]

    for idx, a in enumerate(test_actions):
        # 验证首次动作是否为[0,1]（LL→PM1）
        if idx == 0 and actions[a] != [0, 1]:
            print(f"警告：测试序列第一个动作不是[0,1]，将强制替换为[0,1]以满足首次取片需求")
            a = 0  # 强制第一个动作为[0,1]
        state, reward, done = env.step(a)
        if done:
            break

    # 打印取片序列，验证首次是否为类型1
    print(f"\n最终取片序列（1=类型1，2=类型2）：{env.sequence_log}")
    print(f"首次取片类型：{env.sequence_log[0]}（预期为1）")