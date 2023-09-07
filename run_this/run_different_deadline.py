from run_this.run_methods_and_outputs.run_PPO import *
from run_this.run_methods_and_outputs.run_deadline1 import *
from run_this.run_methods_and_outputs.run_deadline2 import *
from utils import *

with open("./data/all_hitrate.txt", "w") as file:
    file.truncate()

reward, ma_hit_rate, time, ma_all_hit_rate, args = run_ppo();
# 保存all hit ratio曲线
with open("./data/all_hitrate.txt", "a") as file:
    file.write(str(ma_all_hit_rate) + "\n")

reward1, ma_hit_rate1, time1, ma_all_hit_rate1, args1 = run_deadline_1();
# 保存all hit ratio曲线
with open("./data/all_hitrate.txt", "a") as file:
    file.write(str(ma_all_hit_rate1) + "\n")

reward2, ma_hit_rate2, time2, ma_all_hit_rate2, args2 = run_deadline_2();
# 保存all hit ratio曲线
with open("./data/all_hitrate.txt", "a") as file:
    file.write(str(ma_all_hit_rate2) + "\n")

# 绘制不同caching capacity的hitrate对比图
with open("./data/all_hitrate.txt", "r") as file:
    lines = file.readlines()
if len(lines) == 3:
    plot_hit_rate_with_different_deadlines(args, tag='train')
else:
    print("Wrong data size!")
