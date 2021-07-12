# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Usage example:
    taiwan = []  # 统计各省台风登录次数
    fujian = []
    guangdong = []
    zhejiang = []
    year = []
    f = open("Data_pro/t4/typhoon-2010-2019.txt", "r", encoding="utf8")
    for row in list(f)[1:]:
        line = row.split(" ")  # 分割每行数据
        if int(line[0]) < 2010:  # 小于2000年忽略
            continue
        if line[0] not in year:
            year.append(line[0])
            taiwan.append(0)
            fujian.append(0)
            guangdong.append(0)
            zhejiang.append(0)
        else:   # 累加每年登录次数
            index = year.index(line[0])
            if "台湾" in line[5]:
                taiwan[index] += 1
            elif "浙江" in line[5]:
                zhejiang[index] += 1
            elif "福建" in line[5]:
                fujian[index] += 1
            elif "广东" in line[5]:
                guangdong[index] += 1

    data = {
        "TW": taiwan,
        "FJ": fujian,
        "GD": guangdong,
        "ZJ": zhejiang,
    }

    # 显示
    X_axis = np.arange(len(year))
    for i, (name, values) in enumerate(data.items()):
        print(values)
        plt.bar(X_axis - 0.2 * i, values, width=0.2, label=name)

    plt.xticks(X_axis, year)
    plt.xlabel("Year")
    plt.ylabel("Times")
    plt.legend()
    plt.show()
