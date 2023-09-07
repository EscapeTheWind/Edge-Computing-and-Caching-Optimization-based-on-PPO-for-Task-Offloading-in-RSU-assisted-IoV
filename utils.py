#!/usr/bin/env python
# coding=utf-8
"""
@Author: WuCheng
@Email: 1303328276@qq.com
@Date: 2023-6-12 09:20:00
@LastEditor: WuCheng
@LastEditTime: 2023-6-12 09:20:00
Description:
Environment:
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块


def chinese_font():
    """ 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体 """
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_rewards(rewards, cfg, tag='train'):
    plt.xlabel('episodes', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.plot(rewards, label='rewards')
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_rewards_curve.png".format(tag), dpi=1000)
        plt.savefig(cfg.result_path + "{}_rewards_curve.pdf".format(tag), format='pdf', dpi=1000)
    plt.show()


def plot_time(time, cfg, tag='train'):
    plt.xlabel('episodes', fontsize=16)
    plt.ylabel('Objective of computation delay', fontsize=15)
    plt.plot(time, label='time')
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_time_curve.png".format(tag), dpi=1000)
        plt.savefig(cfg.result_path + "{}_time_curve.pdf".format(tag), format='pdf', dpi=1000)
    plt.show()


def plot_hit_rate(ma_hit_rate, cfg, tag='train'):
    plt.xlabel('episodes', fontsize=16)
    plt.ylabel('Objective of content hit rate', fontsize=15)
    plt.plot(ma_hit_rate, label='ma hit rate')
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_hit_rate_curve.png".format(tag), dpi=1000)
        plt.savefig(cfg.result_path + "{}_hit_rate_curve.pdf".format(tag), format='pdf', dpi=1000)
    plt.show()


def plot_all_hit_rate(ma_hit_rate, cfg, tag='train'):
    plt.xlabel('episodes', fontsize=16)
    plt.ylabel('Vehicles\' request response ratio', fontsize=15)
    plt.plot(ma_hit_rate, label='ma hit rate')
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_all_hit_rate_curve.png".format(tag), dpi=1000)
        plt.savefig(cfg.result_path + "{}_all_hit_rate_curve.pdf".format(tag), format='pdf', dpi=1000)
    plt.show()


def plot_hit_rate_with_different_capacities(cfg, tag='train'):
    with open("./data/all_hitrate.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            values = np.fromstring(line[1:-1], sep=', ')
            data.append(values)
    x = np.arange(len(data[0]))
    for i in range(len(data)):
        if i == 0:
            plt.plot(x, data[i], label=f'Capacity = 4 MB')
        elif i == 1:
            plt.plot(x, data[i], label=f'Capacity = 5 MB')
        else:
            plt.plot(x, data[i], label=f'Capacity = 6 MB')
    plt.xlabel('episodes', fontsize=20)
    plt.ylabel('Vehicles\' request response ratio', fontsize=19)
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "different_caching_capacities.png", dpi=1000)
        plt.savefig(cfg.result_path + "different_caching_capacities.pdf", format='pdf',
                    dpi=1000)
    plt.show()


def plot_hit_rate_with_different_deadlines(cfg, tag='train'):
    with open("./data/all_hitrate.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            values = np.fromstring(line[1:-1], sep=', ')
            data.append(values)
    x = np.arange(len(data[0]))
    for i in range(len(data)):
        if i == 0:
            plt.plot(x, data[i], label=f'Deadline = 20')
        elif i == 1:
            plt.plot(x, data[i], label=f'Deadline = 15')
        else:
            plt.plot(x, data[i], label=f'Deadline = 10')
    plt.xlabel('episodes', fontsize=20)
    plt.ylabel('Vehicles\' request response ratio', fontsize=19)
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "different_deadlines.png", dpi=1000)
        plt.savefig(cfg.result_path + "different_deadlines.pdf", format='pdf',
                    dpi=1000)
    plt.show()


def plot_hit_rate_with_different_algo(cfg, tag='train'):
    with open("./data/all_hitrate.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            values = np.fromstring(line[1:-1], sep=', ')
            data.append(values)
    x = np.arange(len(data[0]))
    for i in range(len(data)):
        if i == 0:
            plt.plot(x, data[i], label=f'Our method')
        elif i == 1:
            plt.plot(x, data[i], label=f'LFU')
        elif i == 2:
            plt.plot(x, data[i], label=f'LRU')
    plt.xlabel('episodes', fontsize=20)
    plt.ylabel('Vehicles\' request response ratio', fontsize=19)
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "different_algos1.png", dpi=1000)
        plt.savefig(cfg.result_path + "different_algos1.pdf", format='pdf', dpi=1000)
    plt.show()


def plot_hit_rate_with_different_algo2(cfg, tag='train'):
    with open("./data/all_hitrate.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            values = np.fromstring(line[1:-1], sep=', ')
            data.append(values)
    x = np.arange(len(data[0]))
    for i in range(len(data)):
        if i == 0:
            plt.plot(x, data[i], label=f'Our method')
        if i == 1:
            plt.plot(x, data[i], label=f'Optimal caching')
    plt.xlabel('episodes', fontsize=20)
    plt.ylabel('Vehicles\' request response ratio', fontsize=19)
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "different_algos2.png", dpi=1000)
        plt.savefig(cfg.result_path + "different_algos2.pdf", format='pdf', dpi=1000)
    plt.show()
