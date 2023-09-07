import dataclasses
import numpy as np


@dataclasses.dataclass
class VehicularEnvConfig:
    """参数"""
    """时隙相关"""
    time_slot_start: int = 0
    time_slot_end: int = 199

    """任务相关"""
    min_function_kind: int = 1
    max_function_kind: int = 20

    average_input_datasize: float = 5 * 10  # 5MB
    average_output_datasize: float = 0.5 * 10  # 0.5MB
    average_deadline: int = 12

    start_input_size: float = 5 * 10  # 5MB
    start_difficulty: float = 0.99
    start_output_size: float = 0.5 * 10  # 0.5MB
    start_kind = 1
    start_deadline = 12

    future_weight = 0.9

    """任务队列相关"""
    min_rsu_task_number: int = 6
    max_rsu_task_number: int = 8
    min_vehicle_task_number: int = 1
    max_vehicle_task_number: int = 2
    min_task_data_size: float = 5 * 10  # 5 MB
    max_task_data_size: float = 10 * 10  # 10 MB
    threshold_datasize: float = 1000  # 限制参数

    """车辆队列相关"""
    road_range: int = 500
    min_vehicle_speed: int = 25
    max_vehicle_speed: int = 50

    min_vehicle_compute_speed: float = 3 * 10  # 3M/s
    max_vehicle_compute_speed: float = 6 * 10  # 6M/s

    start_vehicle_number = 5
    vehicle_number = np.random.randint(3, 9, 5001)
    vehicle_seed = 1

    """RSU队列相关"""
    rsu_number = 3
    min_rsu_compute_speed: float = 5 * 10  # 5M/s
    max_rsu_compute_speed: float = 10 * 10  # 10M/s
    cache_capacity: int = 0.5 * 10 * 8  # 4MB

    """传输相关"""
    theta_rr = 1 / 75  # 7.5M/s
    theta_rv = 1 / 50  # 5M/s
    theta_vc = 1 / 2.5  # 0.25M/s
    theta_rc = 1 / 5  # 0.5M/s

    """环境相关"""
    high = np.array([np.finfo(np.float32).max for _ in range(191)])
    low = np.array([0 for _ in range(191)])
    action0_size = 104
    action1_size = 3
    action_size = action0_size * action1_size + 1
