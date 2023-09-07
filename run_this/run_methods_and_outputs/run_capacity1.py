import datetime
import os
import sys
import argparse

from methods.PPO_Discrete import *
from envs import capacity1_environment as environment

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    evaluate_cur_hitrate = 0
    evaluate_all_hitrate = 0
    evaluate_time = 0
    evaluate_none_cache = 0
    evaluate_rsu_offloading_count = 0
    evaluate_vehicle_count = 0
    evaluate_cloud_count = 0
    evaluate_cache_count = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        episode_cur_hitrate = 0
        episode_all_hitrate = 0
        episode_time = 0
        episode_none_cache = 0
        episode_rsu_offloading_count = 0
        episode_vehicle_count = 0
        episode_cloud_count = 0
        episode_cache_count = 0
        i = 0
        while not done:
            i += 1
            vehicle_number = env._get_vehicle_number()
            rsu_number = env._get_rsu_number()
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if a != 0:
                episode_cache_count += 1
                action = [(a - 1) // 3 + 1, (a - 1) % 3 + 1]  # 找出具体的计算节点和缓存节点
                if action[0] == 1:
                    episode_cloud_count += 1
                elif action[0] <= rsu_number:
                    episode_rsu_offloading_count += 1
                elif action[0] <= (rsu_number + vehicle_number):
                    episode_vehicle_count += 1
            s_, r, done, cur_hitrate, non_caching_hitrate, all_hitrate, time = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            episode_cur_hitrate += cur_hitrate
            episode_all_hitrate += all_hitrate
            episode_time += time
            episode_none_cache += non_caching_hitrate
            s = s_
        evaluate_reward += episode_reward
        evaluate_cur_hitrate += episode_cur_hitrate
        evaluate_all_hitrate += episode_all_hitrate
        evaluate_time += episode_time
        evaluate_none_cache += episode_none_cache
        evaluate_rsu_offloading_count += episode_rsu_offloading_count
        evaluate_vehicle_count += episode_vehicle_count
        evaluate_cloud_count += episode_cloud_count
        evaluate_cache_count += episode_cache_count

    return evaluate_reward / times, evaluate_cur_hitrate / times, (
                evaluate_all_hitrate / times) / 100, evaluate_time / times, (
                       evaluate_none_cache / times) / 100, evaluate_rsu_offloading_count / times, evaluate_vehicle_count / times, evaluate_cloud_count / times, evaluate_cache_count / times


def train(args):
    start_time = datetime.datetime.now()
    print('Start training!')
    print(f'Env: {args.env_name}, Algo: {args.algo_name}, Device: {args.device}')
    env = environment.RoadState()
    env_evaluate = environment.RoadState()  # When evaluating the policy, we need to rebuild an environment

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = 500  # Maximum number of steps per episode

    print(f"n states: {args.state_dim}, n actions: {args.action_dim}")

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    episode = 0
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    hit_rate = []
    ma_hit_rate = []
    all_hit_rate = []
    ma_all_hit_rate = []
    time_ = []
    ma_time_ = []
    none_cache_ = []
    rsu_offloading_rate = []
    cloud_offloading_rate = []
    vehicle_offloading_rate = []
    while total_steps < args.max_train_steps:
        episode += 1
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, cur_hitrate, non_caching_hitrate, all_hitrate, time = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, evaluate_cur_hitrate, evaluate_all_hitrate, evaluate_time, evaluate_none_cache, evaluate_rsu_offloading_count, evaluate_vehicle_offloading_count, evaluate_cloud_offloading_count, evaluate_cache_count = evaluate_policy(
                    args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                rewards.append(evaluate_reward)
                hit_rate.append(evaluate_cur_hitrate)
                all_hit_rate.append(evaluate_all_hitrate)
                time_.append(evaluate_time)
                none_cache_.append(evaluate_none_cache)
                rsu_offloading_rate.append(evaluate_rsu_offloading_count)
                vehicle_offloading_rate.append(evaluate_vehicle_offloading_count)
                cloud_offloading_rate.append(evaluate_cloud_offloading_count)
                if ma_rewards:
                    ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * evaluate_reward)
                else:
                    ma_rewards.append(evaluate_reward)
                if ma_hit_rate:
                    ma_hit_rate.append(0.9 * ma_hit_rate[-1] + 0.1 * evaluate_cur_hitrate)
                else:
                    ma_hit_rate.append(evaluate_cur_hitrate)
                if ma_time_:
                    ma_time_.append(0.9 * ma_time_[-1] + 0.1 * evaluate_time)
                else:
                    ma_time_.append(evaluate_time)
                if ma_all_hit_rate:
                    ma_all_hit_rate.append(0.9 * ma_all_hit_rate[-1] + 0.1 * evaluate_all_hitrate)
                else:
                    ma_all_hit_rate.append(evaluate_all_hitrate)
                print(
                    f'Episode: {episode}/{int(args.max_train_steps / args.evaluate_freq)}, Reward: {evaluate_reward:.2f},'
                    f' Hit Rate: {evaluate_cur_hitrate:.2f}, All Hit Rate: {evaluate_all_hitrate:.2f}.')

    end_time = datetime.datetime.now()
    print("start time: " + str(start_time))
    print("end time: " + str(end_time))

    return rewards, ma_hit_rate, time_, ma_all_hit_rate


def run_capacity_1():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument('--algo_name', default='PPO Discrete', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Road State', type=str, help="name of environment")
    parser.add_argument("--max_train_steps", type=int, default=int(100000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=200,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument('--result_path', default=curr_path + \
                                                 "/outputs/" + \
                                                 parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU

    rewards, ma_hit_rate, time_, ma_all_hit_rate = train(args)
    return rewards, ma_hit_rate, time_, ma_all_hit_rate, args
