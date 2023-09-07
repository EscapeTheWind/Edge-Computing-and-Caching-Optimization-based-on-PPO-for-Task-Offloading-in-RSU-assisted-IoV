from run_this.run_methods_and_outputs.run_PPO import *
from utils import *

reward, ma_hit_rate, time, ma_all_hit_rate, args = run_ppo();

# 保存reward曲线
with open("./data/reward.txt", "a") as file:
    file.write(str(reward) + "\n")

# 保存hitrate曲线
with open("./data/hitrate.txt", "a") as file:
    file.write(str(ma_hit_rate) + "\n")

# 保存time曲线
with open("./data/time.txt", "a") as file:
    file.write(str(time) + "\n")

# 保存all hit ratio曲线
with open("./data/all_hitrate.txt", "a") as file:
    file.write(str(ma_all_hit_rate) + "\n")

plot_rewards(reward, args, tag="train")  # 画出结果
plot_hit_rate(ma_hit_rate, args, tag="train")
plot_time(time, args, tag="train")
plot_all_hit_rate(ma_all_hit_rate, args, tag="train")
