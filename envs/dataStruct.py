import numpy as np
from typing import List
import random


class CacheStatus(object):
    """缓存情况"""

    def __init__(self, rsu_number: int, max_function_kind: int):
        self._cache_matrix = [[0] * max_function_kind for _ in range(rsu_number)]

    def update_cache_status_by_one(self, rsu_index: int, task_index: int) -> None:
        self._cache_matrix[rsu_index][task_index] = 1

    def update_cache_status_by_zero(self, rsu_index: int, task_index: int) -> None:
        self._cache_matrix[rsu_index][task_index] = 0

    def get_cache_status(self) -> List[List[int]]:
        return self._cache_matrix

    def get_cache_by_index(self, rsu_index, task_index) -> int:
        return self._cache_matrix[rsu_index][task_index]

    def __getitem__(self, index):
        return self._cache_matrix[index]


class Function(object):
    """任务属性及其操作"""

    def __init__(self, input_size: float, difficulty: float, output_size: float, kind: int, deadline: int) -> None:
        self._input_size = input_size
        self._difficulty = difficulty
        self._output_size = output_size
        self._kind = kind
        self._deadline = deadline

    def get_input_size(self) -> float:
        return float(self._input_size)

    def get_difficulty(self) -> float:
        return float(self._difficulty)

    def get_output_size(self) -> float:
        return float(self._output_size)

    def get_kind(self) -> int:
        return int(self._kind)

    def get_deadline(self) -> int:
        return int(self._deadline)


class TaskList(object):
    """节点上的待执行任务队列属性及操作"""

    def __init__(
            self,
            task_number: int,
            minimum_data_size: float,
            maximum_data_size: float,
            seed: int
    ) -> None:
        self._task_number = task_number
        self._minimum_data_size = minimum_data_size
        self._maximum_data_size = maximum_data_size
        self._seed = seed

        # 生成每个任务的数据量大小
        np.random.seed(self._seed)
        self._data_sizes = np.random.uniform(self._minimum_data_size, self._maximum_data_size, self._task_number)

        self._task_list = [_ for _ in self._data_sizes]

    def get_task_list(self) -> List[float]:
        return self._task_list

    def sum_data_size(self) -> float:
        """返回该节点的总任务量"""
        return sum(self._task_list)

    def add_task_list(self, new_data_size) -> None:
        """如果卸载到该节点，任务队列会增加"""
        self._task_list.append(new_data_size)

    def add_by_slot(self, task_number) -> None:
        data_sizes = np.random.uniform(self._minimum_data_size, self._maximum_data_size, task_number)
        for datasize in data_sizes:
            self._task_list.append(datasize)
            self._task_number += 1

    def delete_data_list(self, process_ability) -> None:
        """在时间转移中对车辆任务队列进行处理"""
        while True:
            # 如果队列中没有任务
            if len(self._task_list) == 0:
                break
            # 如果队列中有任务
            elif process_ability >= self._task_list[0]:  # 单位时间计算能力大于数据量
                process_ability -= self._task_list[0]
                del self._task_list[0]
            else:  # 单位时间计算能力小于数据量
                self._task_list[0] -= process_ability
                break


class Vehicle(object):
    """车辆属性及其操作"""

    def __init__(
            self,
            road_range: int,
            vehicle_speed: float,
            min_task_number: float,
            max_task_number: float,
            max_task_data_size: float,
            min_task_data_size: float,
            max_vehicle_compute_speed: float,
            min_vehicle_compute_speed: float,
            seed: int
    ) -> None:
        # 车辆在场景中的生存时间生成
        self._road_range = road_range
        self._vehicle_speed = vehicle_speed
        self._stay_time = int(self._road_range / self._vehicle_speed)

        # 车辆计算速度生成
        self._max_compute_speed = max_vehicle_compute_speed
        self._min_compute_speed = min_vehicle_compute_speed
        self._seed = seed
        np.random.seed(self._seed)
        self._compute_speed = np.random.uniform(self._min_compute_speed, self._max_compute_speed, 1)

        # 车辆任务队列生成
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_data_size = max_task_data_size
        self._min_data_size = min_task_data_size
        np.random.seed(self._seed)
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._vehicle_task_list = TaskList(self._task_number, self._min_data_size, self._max_data_size, self._seed)

    # 生存时间相关
    def get_stay_time(self) -> int:
        return self._stay_time

    def decrease_stay_time(self) -> int:
        self._stay_time -= 1
        return self._stay_time

    def is_out(self) -> bool:
        if self._stay_time <= 5:  # 快要出去的车辆或者速度很大的车辆对任务不会感兴趣
            return True
        else:
            return False

    # 车辆速度相关
    def get_vehicle_compute_speed(self) -> float:
        return self._compute_speed

    # 车辆任务队列相关
    def get_task_list(self) -> TaskList:
        return self._vehicle_task_list

    def get_sum_tasks(self) -> float:
        if len(self._vehicle_task_list.get_task_list()) == 0:  # 车辆上没有任务
            return 0
        else:
            return self._vehicle_task_list.sum_data_size()  # 车辆上有任务

    def generate_request(self):
        weights = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        probability = 0.95  # 调整这个概率值以控制请求的频率
        if random.random() < probability:
            task_type = np.random.choice(range(1, 21), p=weights / np.sum(weights))  # 用于指定请求的任务类型
            return task_type
        else:
            return None


class VehicleList(object):
    """实现场景中车辆的管理，包括车辆更新、停留时间更新以及任务队列更新"""

    def __init__(
            self,
            vehicle_number: int,
            road_range: int,
            min_vehicle_speed: float,
            max_vehicle_speed: float,
            min_task_number: float,
            max_task_number: float,
            min_task_data_size: float,
            max_task_data_size: float,
            min_vehicle_compute_speed: float,
            max_vehicle_compute_speed: float,
            seed: int
    ) -> None:
        self._seed = seed
        self._vehicle_number = vehicle_number
        self._road_range = road_range
        self._min_vehicle_speed = min_vehicle_speed
        self._max_vehicle_speed = max_vehicle_speed
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_data_size = min_task_data_size
        self._max_data_size = max_task_data_size
        self._min_compute_speed = min_vehicle_compute_speed
        self._max_compute_speed = max_vehicle_compute_speed

        self._vehicle_speed = np.random.uniform(self._min_vehicle_speed, self._max_vehicle_speed)

        self.vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                vehicle_speed=self._vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                max_task_data_size=self._max_data_size,
                min_task_data_size=self._min_data_size,
                max_vehicle_compute_speed=self._max_compute_speed,
                min_vehicle_compute_speed=self._min_compute_speed,
                seed=self._seed)
            for _ in range(self._vehicle_number)]

    def get_vehicle_number(self) -> int:
        """返回车辆数量"""
        return self._vehicle_number

    def get_vehicle_list(self) -> List[Vehicle]:
        """返回车辆队列"""
        return self.vehicle_list

    def add_stay_vehicle(self, new_vehicle_number) -> None:
        """随时间增加车辆数量"""
        np.random.seed(self._seed)
        new_vehicle_list = [
            Vehicle(
                road_range=self._road_range,
                vehicle_speed=self._vehicle_speed,
                min_task_number=self._min_task_number,
                max_task_number=self._max_task_number,
                max_task_data_size=self._max_data_size,
                min_task_data_size=self._min_data_size,
                max_vehicle_compute_speed=self._max_compute_speed,
                min_vehicle_compute_speed=self._min_compute_speed,
                seed=self._seed)
            for _ in range(new_vehicle_number)]

        self.vehicle_list = self.vehicle_list + new_vehicle_list
        self._vehicle_number += new_vehicle_number

    def delete_out_vehicle(self) -> None:
        """从队列中删除不在范围内的车辆"""
        i = 0
        while i < len(self.vehicle_list):
            if len(self.vehicle_list) == 0:
                pass
            elif self.vehicle_list[i].is_out():
                del self.vehicle_list[i]
                self._vehicle_number -= 1
            else:
                i += 1


class RSU(object):
    """RSU"""

    def __init__(
            self,
            max_task_number: float,
            min_task_number: float,
            max_task_data_size: float,
            min_task_data_size: float,
            max_rsu_compute_speed: float,
            min_rsu_compute_speed: float,
            cache_capacity: float,
            seed: int
    ) -> None:
        # rsu计算速度生成
        self._max_compute_speed = max_rsu_compute_speed
        self._min_compute_speed = min_rsu_compute_speed
        self._seed = seed
        np.random.seed(self._seed)
        self._compute_speed = np.random.uniform(self._min_compute_speed, self._max_compute_speed, 1)

        # 车辆任务队列生成
        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._max_data_size = max_task_data_size
        self._min_data_size = min_task_data_size
        np.random.seed(self._seed)
        self._task_number = np.random.randint(self._min_task_number, self._max_task_number)
        self._rsu_task_list = TaskList(self._task_number, self._min_data_size, self._max_data_size, self._seed)

        # 缓存队列生成
        self.cache_capacity = cache_capacity
        self._cache_list = CacheList()

    def get_rsu_compute_speed(self) -> float:
        return self._compute_speed

    def get_task_list(self) -> TaskList:
        return self._rsu_task_list

    def get_sum_tasks(self) -> float:
        if len(self._rsu_task_list.get_task_list()) == 0:  # RSU上没有任务
            return 0
        else:
            return self._rsu_task_list.sum_data_size()  # RSU上有任务

    def store_task_with_popularity(self, output_data: float, kind: int, deadline: int, popularity: float, last_request: int, frequency: int) -> List[int]:
        re = []
        if self._cache_list.is_category_present(kind):
            self._cache_list.remove_category(kind)
        while self._cache_list.get_used_space() + output_data > self.cache_capacity:
            a = self._cache_list.remove_least_popular_task()
            re.append(a)
        self._cache_list.add_task(output_data, kind, deadline, popularity, last_request, frequency)
        return re

    def store_task_with_request(self, output_data: float, kind: int, deadline: int, popularity: float, last_request: int, frequency: int) -> List[int]:
        re = []
        if self._cache_list.is_category_present(kind):
            self._cache_list.remove_category(kind)
        while self._cache_list.get_used_space() + output_data > self.cache_capacity:
            a = self._cache_list.remove_least_recent_task()
            re.append(a)
        self._cache_list.add_task(output_data, kind, deadline, popularity, last_request, frequency)
        return re

    def store_task_with_frequency(self, output_data: float, kind: int, deadline: int, popularity: float, last_request: int, frequency: int) -> List[int]:
        re = []
        if self._cache_list.is_category_present(kind):
            self._cache_list.remove_category(kind)
        while self._cache_list.get_used_space() + output_data > self.cache_capacity:
            a = self._cache_list.remove_least_frequent_task()
            re.append(a)
        self._cache_list.add_task(output_data, kind, deadline, popularity, last_request, frequency)
        return re

    def delete_task(self, task_info: int) -> None:
        self._cache_list.remove_category(task_info)


class CacheList(object):
    """每个RSU上的缓存空间"""

    def __init__(self):
        self._cache_list = []

    def add_task(self, output_data: float, kind: int, deadline: int, popularity: float, last_request: int, frequency: int) -> None:
        self._cache_list.append((output_data, kind, deadline, popularity, last_request, frequency))

    def remove_least_popular_task(self) -> int:
        self._cache_list.sort(key=lambda task: task[3])
        re = self._cache_list[0][1]
        self._cache_list.pop(0)
        return re

    def remove_least_recent_task(self) -> int:
        self._cache_list.sort(key=lambda task: task[4])
        re = self._cache_list[0][1]
        self._cache_list.pop(0)
        return re

    def remove_least_frequent_task(self) -> int:
        self._cache_list.sort(key=lambda task: task[5])
        re = self._cache_list[0][1]
        self._cache_list.pop(0)
        return re

    def remove_category(self, kind: int) -> None:
        self._cache_list = [task for task in self._cache_list if task[1] != kind]

    def get_used_space(self) -> float:
        return sum(task[0] for task in self._cache_list)

    def is_category_present(self, kind: int) -> bool:
        return any(task[1] == kind for task in self._cache_list)

    def get_cache_list(self):
        return self._cache_list

    def set_cache_list(self, cache_list: tuple):
        self._cache_list.clear()
        for cache_element in cache_list:
            self._cache_list.append(cache_element)
        return self._cache_list

class RSUList(object):
    """RSU队列"""

    def __init__(
            self,
            rsu_number,
            max_task_number: float,
            min_task_number: float,
            max_task_data_size: float,
            min_task_data_size: float,
            max_rsu_compute_speed: float,
            min_rsu_compute_speed: float,
            cache_capacity: int,
            seed: int
    ) -> None:
        self._seed = seed
        self._rsu_number = rsu_number

        self._min_task_number = min_task_number
        self._max_task_number = max_task_number
        self._min_data_size = min_task_data_size
        self._max_data_size = max_task_data_size
        self._min_compute_speed = min_rsu_compute_speed
        self._max_compute_speed = max_rsu_compute_speed
        self.cache_capacity = cache_capacity

        self._rsu_list = [
            RSU(
                max_task_number=self._max_task_number,
                min_task_number=self._min_task_number,
                max_task_data_size=self._max_data_size,
                min_task_data_size=self._min_data_size,
                max_rsu_compute_speed=self._max_compute_speed,
                min_rsu_compute_speed=self._min_compute_speed,
                seed=self._seed,
                cache_capacity=self.cache_capacity
            )
            for _ in range(rsu_number)]

    def get_rsu_number(self):
        return self._rsu_number

    def get_rsu_list(self):
        return self._rsu_list


class TimeSlot(object):
    """时隙属性及操作"""

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.slot_length = self.end - self.start

        self.now = start
        self.reset()

    def __str__(self):
        return f"now time: {self.now}, [{self.start} , {self.end}] with {self.slot_length} slots"

    def add_time(self) -> None:
        """add time to the system"""
        self.now += 1

    def is_end(self) -> bool:
        """check if the system is at the end of the time slots"""
        return self.now >= self.end

    def get_slot_length(self) -> int:
        """get the length of each time slot"""
        return self.slot_length

    def get_now(self) -> int:
        return self.now

    def reset(self) -> None:
        self.now = self.start
