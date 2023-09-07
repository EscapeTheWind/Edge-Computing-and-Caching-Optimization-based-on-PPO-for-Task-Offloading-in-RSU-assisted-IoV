import copy
import random
import time
from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import RenderFrame, ObsType

from envs import LFU_config
from envs.dataStruct import VehicleList, RSUList, TimeSlot, Function, CacheStatus


class RoadState(gym.Env):
    """updating"""

    def __init__(
            self,
            env_config: Optional[LFU_config.VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None,
            function: Optional[Function] = None,
            cache_status: Optional[CacheStatus] = None,
    ):
        self.request_history = []
        self.popularity = [0.0] * 20
        self.last_popularity = self.popularity
        self.last_request = [0] * 20
        self.frequency = [0] * 20
        self.road_heat = [random.uniform(0, 0.1),
             random.uniform(0.2, 0.3),
             random.uniform(0.8, 1)]
        total = sum(self.road_heat)
        self.road_heat = [heat / total for heat in self.road_heat]
        self.last_road_heat = self.road_heat

        if env_config is None:
            self._config = LFU_config.VehicularEnvConfig()
        else:
            self._config = env_config

        if time_slot is None:
            self._timeslot: TimeSlot = TimeSlot(start=self._config.time_slot_start, end=self._config.time_slot_end)
        else:
            self._timeslot = time_slot

        if cache_status is None:
            self._cache_matrix = CacheStatus(rsu_number=self._config.rsu_number,
                                             max_function_kind=self._config.max_function_kind)

        if function is None:
            self._function: Function = Function(
                input_size=self._config.start_input_size,
                difficulty=self._config.start_difficulty,
                output_size=self._config.start_output_size,
                kind=self._config.start_kind,
                deadline=self._config.start_deadline
            )
        else:
            self._function = function

        if vehicle_list is None:
            self._vehicle_list: VehicleList = VehicleList(
                vehicle_number=self._config.start_vehicle_number,
                road_range=self._config.road_range,
                min_vehicle_speed=self._config.min_vehicle_speed,
                max_vehicle_speed=self._config.max_vehicle_speed,
                max_vehicle_compute_speed=self._config.max_vehicle_compute_speed,
                min_vehicle_compute_speed=self._config.min_vehicle_compute_speed,
                max_task_number=self._config.max_vehicle_task_number,
                min_task_number=self._config.min_vehicle_task_number,
                max_task_data_size=self._config.max_task_data_size,
                min_task_data_size=self._config.min_task_data_size,
                seed=self._config.vehicle_seed
            )
        else:
            self._vehicle_list = vehicle_list

        if rsu_list is None:
            self._rsu_list: RSUList = RSUList(
                rsu_number=self._config.rsu_number,
                max_task_number=self._config.max_rsu_task_number,
                min_task_number=self._config.min_rsu_task_number,
                max_task_data_size=self._config.max_task_data_size,
                min_task_data_size=self._config.min_task_data_size,
                max_rsu_compute_speed=self._config.max_rsu_compute_speed,
                min_rsu_compute_speed=self._config.min_rsu_compute_speed,
                seed=self._config.vehicle_seed,
                cache_capacity=self._config.cache_capacity
            )
        else:
            self._rsu_list = rsu_list

        self.action_space = spaces.Discrete(self._config.action_size)
        self.observation_space = spaces.Box(low=self._config.low, high=self._config.high, dtype=np.float32)

        self.state = None
        self.seed = int(time.time())
        self.reward = 0

        self.last_hit_reward = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        self._timeslot.reset()  # 重置时间

        self._cache_matrix = CacheStatus(rsu_number=self._config.rsu_number,
                                         max_function_kind=self._config.max_function_kind)
        self.request_history = []
        self.popularity = [0.0] * self._config.max_function_kind
        self.last_popularity = self.popularity
        self.last_request = [0] * self._config.max_function_kind
        self.frequency = [0] * self._config.max_function_kind
        self.road_heat = [random.uniform(0, 0.1),
             random.uniform(0.2, 0.3),
             random.uniform(0.8, 1)]
        total = sum(self.road_heat)
        self.road_heat = [heat / total for heat in self.road_heat]
        self.last_road_heat = self.road_heat

        self._function = Function(
            input_size=self._config.start_input_size,
            difficulty=self._config.start_difficulty,
            output_size=self._config.start_output_size,
            kind=self._config.start_kind,
            deadline=self._config.start_deadline
        )
        self._rsu_list = RSUList(
            rsu_number=self._config.rsu_number,
            max_task_number=self._config.max_rsu_task_number,
            min_task_number=self._config.min_rsu_task_number,
            max_task_data_size=self._config.max_task_data_size,
            min_task_data_size=self._config.min_task_data_size,
            max_rsu_compute_speed=self._config.max_rsu_compute_speed,
            min_rsu_compute_speed=self._config.min_rsu_compute_speed,
            seed=self._config.vehicle_seed,
            cache_capacity=self._config.cache_capacity
        )
        self._vehicle_list = VehicleList(
            vehicle_number=self._config.start_vehicle_number,
            road_range=self._config.road_range,
            min_vehicle_speed=self._config.min_vehicle_speed,
            max_vehicle_speed=self._config.max_vehicle_speed,
            max_vehicle_compute_speed=self._config.max_vehicle_compute_speed,
            min_vehicle_compute_speed=self._config.min_vehicle_compute_speed,
            max_task_number=self._config.max_vehicle_task_number,
            min_task_number=self._config.min_vehicle_task_number,
            max_task_data_size=self._config.max_task_data_size,
            min_task_data_size=self._config.min_task_data_size,
            seed=self._config.vehicle_seed
        )
        obs_1 = [
            self._function.get_input_size(),
            self._function.get_difficulty(),
            self._function.get_output_size(),
            self._function.get_kind(),
            self._function.get_deadline()
        ]
        obs_2 = []
        for rsu in self._rsu_list.get_rsu_list():
            obs_2.append(rsu.get_sum_tasks())
        obs_3 = []
        for vehicle in self._vehicle_list.get_vehicle_list():
            obs_3.append(vehicle.get_sum_tasks())
        obs_4 = []
        for popularity in self.popularity:
            obs_4.append(popularity)
        obs_5 = []
        for i in self._cache_matrix:
            for j in i:
                obs_5.append(j)
        obs_6 = []
        for heat in self.road_heat:
            obs_6.append(heat)
        obs_7 = []
        for _ in range(104 - self._vehicle_list.get_vehicle_number()
                       - self._rsu_list.get_rsu_number() - 1):
            obs_7.append(self._config.threshold_datasize)

        self.state = obs_1 + obs_2 + obs_3 + obs_4 + obs_5 + obs_6 + obs_7

        return np.array(self.state)

    def _get_vehicle_number(self) -> int:
        return self._vehicle_list.get_vehicle_number()

    def _get_rsu_number(self) -> int:
        return self._rsu_list.get_rsu_number()

    def _function_generation(self, min_kind, max_kind):
        self.seed += self._timeslot.get_now()
        np.random.seed(self.seed)
        input_data = int(np.random.normal(self._config.average_input_datasize))
        np.random.seed(self.seed)
        difficulty = float(np.random.randint(8, 10) / 10)
        np.random.seed(self.seed)
        output_data = int(np.random.normal(self._config.average_output_datasize))
        kind = np.random.randint(min_kind, max_kind + 1)
        deadline = int(np.random.normal(self._config.average_deadline))

        self._function = Function(input_data, difficulty, output_data, kind, deadline)
        return self._function

    def _function_cache_allocation(self, action) -> None:
        """放置缓存任务"""
        action_arr = [(action - 1) // 3 + 1, (action - 1) % 3 + 1]  # 找出具体的计算节点和缓存节点

        if action_arr[0] <= 4 + self._vehicle_list.get_vehicle_number():
            i = 0
            for rsu in self._rsu_list.get_rsu_list():
                # 检查每个RSU是否存在kind类任务,如果存在，则先删除该任务
                if rsu._cache_list.is_category_present(self._function.get_kind()):
                    rsu.delete_task(self._function.get_kind())
                    self._cache_matrix.update_cache_status_by_zero(i, self._function.get_kind() - 1)
                i += 1
            re = self._rsu_list.get_rsu_list()[action_arr[1] - 1].store_task_with_frequency(
                self._function.get_output_size(),
                self._function.get_kind(),
                self._function.get_deadline(),
                self.popularity[self._function.get_kind() - 1],
                self.last_request[self._function.get_kind() - 1],
                self.frequency[self._function.get_kind() - 1])
            self._cache_matrix.update_cache_status_by_one(action_arr[1] - 1, self._function.get_kind() - 1)
            if re:  # 存入任务时如果因空间不足删除了其他任务，将其他任务的缓存情况置0
                for task_info in re:
                    self._rsu_list.get_rsu_list()[action_arr[1] - 1].delete_task(task_info)
                    self._cache_matrix.update_cache_status_by_zero(action_arr[1] - 1, task_info - 1)
        else:
            pass

    def _function_compute_allocation(self, action) -> None:
        """放置计算任务"""
        action_arr = [(action - 1) // 3 + 1, (action - 1) % 3 + 1]  # 找出具体的计算节点和缓存节点
        if action_arr[0] == 1:
            pass
        elif action_arr[0] <= 4:
            self._rsu_list.get_rsu_list()[action_arr[0] - 2].get_task_list().add_task_list(
                self._function.get_input_size())
        elif action_arr[0] <= (4 + self._vehicle_list.get_vehicle_number()):
            self._vehicle_list.get_vehicle_list()[action_arr[0] - 5].get_task_list().add_task_list(
                self._function.get_input_size())
        else:
            pass

    def _update_road(self, action) -> object:
        """更新道路状态"""
        # 增加时隙
        self._timeslot.add_time()
        now = self._timeslot.get_now()

        # 更新车流量热度
        self.last_road_heat = self.road_heat
        self.road_heat = [random.uniform(0, 0.1),
             random.uniform(0.2, 0.3),
             random.uniform(0.8, 1)]
        total = sum(self.road_heat)
        self.road_heat = [heat / total for heat in self.road_heat]

        # 自动删除到期任务
        for rsu in self._rsu_list.get_rsu_list():
            if rsu._cache_list.get_cache_list():
                cache_list = list(rsu._cache_list.get_cache_list())  # 将元组转换为列表
                new_cache_list = []
                for task in cache_list:
                    task_list = list(task)  # 将元组中的元素转换为列表
                    task_list[2] -= 1  # 修改任务的截止时间
                    if task_list[2] > 0:
                        new_cache_list.append(tuple(task_list))  # 将仍存在任务转换回元组并放回列表中
                    else:
                        self._cache_matrix.update_cache_status_by_zero(0, task_list[1] - 1)  # 将该类到期任务从所有RSU中删除
                        self._cache_matrix.update_cache_status_by_zero(1, task_list[1] - 1)
                        self._cache_matrix.update_cache_status_by_zero(2, task_list[1] - 1)
                rsu._cache_list.set_cache_list(tuple(new_cache_list))

        if action != 0:
            # 放置缓存任务
            self._function_cache_allocation(action)
            # 放置计算任务
            self._function_compute_allocation(action)

        # 更新任务队列
        for vehicle in self._vehicle_list.get_vehicle_list():
            vehicle.decrease_stay_time()
            process_ability = copy.deepcopy(vehicle.get_vehicle_compute_speed())
            vehicle.get_task_list().delete_data_list(process_ability)
            vehicle.get_task_list().add_by_slot(1)
            request = vehicle.generate_request()  # 队列中的每辆车有几率生成任务请求
            if request is not None:
                self.last_request[request - 1] = now
                self.frequency[request - 1] += 1
                self.request_history.append([now, request])  # 将任务请求加入历史记录

        for rsu in self._rsu_list.get_rsu_list():
            process_ability = copy.deepcopy(rsu.get_rsu_compute_speed())
            rsu.get_task_list().delete_data_list(process_ability)
            rsu.get_task_list().add_by_slot(2)

        # 根据Hawkes计算流行度
        phi = 0.9
        delta = 0.1
        counts = np.zeros(20)
        history_length = min(len(self.request_history), 100)  # 最多统计100条请求
        current_history = self.request_history[-history_length:]
        for d in current_history:
            counts[d[1] - 1] += 1
        total_count = np.sum(counts)  # 总计数

        self.last_popularity = self.popularity
        if total_count == 0:
            self.popularity = np.zeros(20)
        else:
            for i in range(0, 20):
                popularity = 0
                for v in current_history:
                    if v[1] - 1 == i:
                        dt = self._timeslot.get_now() - v[0]
                        popularity += np.exp(-delta * dt)
                self.popularity[i] = self.last_popularity[i] + phi * popularity
            self.popularity /= np.sum(self.popularity)

        # 判断是否要删除车辆
        self._vehicle_list.delete_out_vehicle()

        # 更新车辆
        if self._config.vehicle_number[now] == 0:
            pass
        elif self._vehicle_list.get_vehicle_number() + self._config.vehicle_number[now] > 100:
            pass
        else:
            self._vehicle_list.add_stay_vehicle(self._config.vehicle_number[now])

        return self._vehicle_list, self._rsu_list

    def get_time_reward(self, action):
        if action == 0:
            return 0
        action_arr = [(action - 1) // 3 + 1, (action - 1) % 3 + 1]  # 找出具体的计算节点和缓存节点

        if action_arr[0] == 1:  # 卸载到Cloud
            upload_time = self.state[0] * self._config.theta_rc
            wait_time = 0
            compute_time = 0
            download_time = self.state[2] * self._config.theta_rc
            total_time = upload_time + wait_time + compute_time + download_time
            self.reward = - total_time
        elif action_arr[0] <= 4:  # 卸载到RSU
            upload_time = self.state[0] * self._config.theta_rr
            wait_time = self.state[action_arr[0] + 5] / self._rsu_list.get_rsu_list()[
                action_arr[0] - 2].get_rsu_compute_speed()
            compute_time = self.state[0] / (
                    self.state[1] * self._rsu_list.get_rsu_list()[action_arr[0] - 2].get_rsu_compute_speed())
            download_time = self._config.theta_rr * self.state[2]
            total_time = upload_time + wait_time + compute_time + download_time
            self.reward = - total_time
        elif action_arr[0] <= (self._vehicle_list.get_vehicle_number() + 4):  # 卸载到Vehicle
            upload_time = self._config.theta_rv * self.state[0]
            wait_time = self.state[action_arr[0] + 5] / self._vehicle_list.get_vehicle_list()[
                action_arr[0] - 5].get_vehicle_compute_speed()
            compute_time = self.state[0] / (
                    self.state[1] * self._vehicle_list.get_vehicle_list()[
                action_arr[0] - 5].get_vehicle_compute_speed())
            download_time = self._config.theta_rv * self.state[2]
            total_time = upload_time + wait_time + compute_time + download_time
            self.reward = - total_time
        else:
            self.reward = -200
        if self.reward < -self.state[4]:  # 如果超过了时间要求
            self.reward = -200
        return float(self.reward)

    def get_future_matrix(self, time, new_matrix, new_cachelist):
        if time == 1:
            # 创建一个新的矩阵对象并将原矩阵的值复制到新对象中
            new_matrix = copy.deepcopy(self._cache_matrix)
            new_cachelist.append(list(copy.deepcopy(self._rsu_list.get_rsu_list()[0]._cache_list.get_cache_list())))
            new_cachelist.append(list(copy.deepcopy(self._rsu_list.get_rsu_list()[1]._cache_list.get_cache_list())))
            new_cachelist.append(list(copy.deepcopy(self._rsu_list.get_rsu_list()[2]._cache_list.get_cache_list())))

        # 自动删除到期任务
        for i in range(0, 3):
            if new_cachelist[i]:
                cache_list = new_cachelist[i]  # 将元组转换为列表
                _new_cache_list = []
                for task in cache_list:
                    task_list = list(task)  # 将元组中的元素转换为列表
                    task_list[2] -= 1  # 修改任务的截止时间
                    if task_list[2] > 0:
                        new_cachelist.append(tuple(task_list))  # 将仍存在任务转换回元组并放回列表中
                    else:
                        new_matrix.update_cache_status_by_zero(0, task_list[1] - 1)  # 将该类到期任务从所有RSU中删除
                        new_matrix.update_cache_status_by_zero(1, task_list[1] - 1)
                        new_matrix.update_cache_status_by_zero(2, task_list[1] - 1)
                new_cachelist[i] = tuple(_new_cache_list)
        return new_matrix, new_cachelist

    def get_all_hitrate_reward(self):
        hitrate_reward = 0
        # 当前时刻的命中率（总）
        for k in range(0, 20):
            for r in range(0, 3):
                hitrate_reward += self.last_popularity[k] * self._cache_matrix[r][k] * self.last_road_heat[r]
        return hitrate_reward

    def get_current_hitrate_reward(self, action):
        if action == 0:
            return 0
        action_arr = [(action - 1) // 3 + 1, (action - 1) % 3 + 1]  # 找出具体的计算节点和缓存节点

        if action_arr[0] <= 4 + self._vehicle_list.get_vehicle_number():
            hitrate_reward = 0
            # 当前时刻的命中率
            for r in range(0, 3):
                hitrate_reward += self.last_popularity[self._function.get_kind() - 1] * self._cache_matrix[r][
                    self._function.get_kind() - 1] * self.last_road_heat[r] * self._get_vehicle_number()
            # 未来第一个时刻的命中率
            new_matrix, new_cachelist = self.get_future_matrix(1, [], [])
            for r in range(0, 3):
                hitrate_reward += self.popularity[self._function.get_kind() - 1] * new_matrix[r][
                    self._function.get_kind() - 1] * self._config.future_weight * self.road_heat[
                                      r] * self._get_vehicle_number()
            # 未来第二、三、四、五个时刻的命中率
            for i in range(0, 4):
                new_matrix, new_cachelist = self.get_future_matrix(0, new_matrix, new_cachelist)
                for r in range(0, 3):
                    hitrate_reward += self.popularity[self._function.get_kind() - 1] * new_matrix[r][
                        self._function.get_kind() - 1] * self._config.future_weight ** (i + 2) * self.road_heat[
                                          r] * self._get_vehicle_number()
        else:
            hitrate_reward = 0
        return hitrate_reward

    def get_non_caching_hitrate_reward(self):
        non_caching_hitrate_reward = 0
        for k in range(0, 20):
            if self._function.get_kind() == k:
                hit = 1
            else:
                hit = 0
            non_caching_hitrate_reward += self.popularity[k] * hit
        return non_caching_hitrate_reward

    def step(self, action):
        time_reward = self.get_time_reward(action)

        # done更新
        done = self._timeslot.is_end()

        # 状态更新
        function = self._function_generation(self._config.min_function_kind, self._config.max_function_kind)

        self._vehicle_list, self._rsu_list = self._update_road(action)
        hitrate_reward = self.get_current_hitrate_reward(action)
        all_hitrate_reward = self.get_all_hitrate_reward()
        # 奖励值更新
        reward = 0.01 * time_reward + 0.99 * hitrate_reward

        obs_p1 = [function.get_input_size(), function.get_difficulty(), function.get_output_size(),
                  function.get_kind(), function.get_deadline()]
        obs_p2 = []
        for rsu in self._rsu_list.get_rsu_list():
            obs_p2.append(rsu.get_sum_tasks())
        obs_p3 = []
        for vehicle in self._vehicle_list.get_vehicle_list():
            obs_p3.append(vehicle.get_sum_tasks())
        obs_p4 = []
        for popularity in self.popularity:
            obs_p4.append(popularity)
        obs_p5 = []
        for i in self._cache_matrix:
            for j in i:
                obs_p5.append(j)
        obs_p6 = []
        for heat in self.road_heat:
            obs_p6.append(heat)
        obs_p7 = []
        if 104 - self._vehicle_list.get_vehicle_number() - self._rsu_list.get_rsu_number() - 1 > 0:
            for i in range(104 - self._vehicle_list.get_vehicle_number()
                           - self._rsu_list.get_rsu_number() - 1):
                obs_p7.append(self._config.threshold_datasize)
        self.state = obs_p1 + obs_p2 + obs_p3 + obs_p4 + obs_p5 + obs_p6 + obs_p7

        return np.array(self.state, dtype=np.float32), reward, done, self.get_current_hitrate_reward(
            action), self.get_non_caching_hitrate_reward(), all_hitrate_reward, self.get_time_reward(action)

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass
