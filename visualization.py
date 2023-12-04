import os
import pandas as pd
import matplotlib.pyplot as plt


bigbench_path = "results/bigbench/"
glue_path = "results/glue/"

tasks_bigbench = ['elementary_math_qa','cryptonite','intersect_geometry','list_functions', 'tracking_shuffled_objects']
tasks_glue = ['mnli','qnli','sst2','qqp']
import numpy as np

# avg = path + "_avg.csv"
# cv = path + "_cv.csv"
# cv_weights = path + "_cv_weights.csv"
# lorahub = path + "_lorahub.csv"
# lorahub_weights = path + "_lorahub_weights.csv"
# single = path + "_single.csv"

# class Model:
#     # 类的构造函数（初始化方法）
#     def __init__(self, name):
#         self.task = task  # 类的属性
#         self.avg = pd.read_csv(avg, header=None)
#         self.cv = pd.read_csv(cv, header=None)
#         self.cv_weights = pd.read_csv(cv_weights, header=None)
#         self.lorahub = pd.read_csv(lorahub, header=None)
#         self.lorahub_weights = pd.read_csv(lorahub_weights, header=None)
#         self.single = pd.read_csv(single, header=None)

def get_results_csv(path):
    avg = path + "_avg.csv"
    cv = path + "_cv.csv"
    cv_weights = path + "_cv_weights.csv"
    lorahub = path + "_lorahub.csv"
    lorahub_weights = path + "_lorahub_weights.csv"
    single = path + "_single.csv"
    files = [avg, cv, cv_weights, lorahub, lorahub_weights, single]
    results = []
    for file in files:
        result = pd.read_csv(file, header=None)
        results.append(result)
    return results

results_on_glue = []
for task in tasks_glue:
    results_on_glue.append(get_results_csv(glue_path + task))

results_on_bigbench = []
for task in tasks_bigbench:
    results_on_bigbench.append(get_results_csv(bigbench_path + task))

mean_perf_on_other_clients_bigbench = []
for task in results_on_bigbench:
    lora_perf = task[3]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            if k == j:
                continue
            sum += lora_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    mean_perf_on_other_clients_bigbench.append(client_perf)

mean_perf_on_other_clients_glue = []
for task in results_on_glue:
    lora_perf = task[3]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += lora_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    mean_perf_on_other_clients_glue.append(client_perf)

baseline_glue_single = []
for task in results_on_glue:
    single_perf = task[5]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += single_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    baseline_glue_single.append(client_perf)


baseline_glue_cv = []
for task in results_on_glue:
    cv_perf = task[1]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += cv_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    baseline_glue_cv.append(client_perf)

baseline_glue_avg = []
for task in results_on_glue:
    avg_perf = task[0][3:-2]
    baseline_glue_avg.append(avg_perf)


baseline_bigbench_single = []
for task in results_on_bigbench:
    single_perf = task[5]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            if k == j:
                continue
            sum += single_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    baseline_bigbench_single.append(client_perf)

baseline_bigbench_cv = []
for task in results_on_bigbench:
    cv_perf = task[1]
    client_perf = []
    for j in range(3, 10):
        sum = 0
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += cv_perf[j][k]
        sum /= 7
        client_perf.append(sum)
    baseline_bigbench_cv.append(client_perf)

baseline_bigbench_avg = []
for task in results_on_bigbench:
    avg_perf = task[0][3:-2]
    baseline_bigbench_avg.append(avg_perf)



for i in range(len(mean_perf_on_other_clients_glue)):
    print(mean_perf_on_other_clients_glue[i])
fig, ax = plt.subplots()
# 绘制折线图
plt.plot(mean_perf_on_other_clients_glue[0], label='mnli', color='blue')
plt.plot(mean_perf_on_other_clients_glue[1], label='qnli', color='green')
plt.plot(mean_perf_on_other_clients_glue[2], label='qqp', color='red')
plt.plot(mean_perf_on_other_clients_glue[3], label='sst2', color='purple')

# 添加图例
plt.legend()

# 设置X轴刻度和标签
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # X轴刻度的值
x_labels = ['Cli-1', 'Cli_2', 'Cli_3', 'Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)



ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)

# 添加标题和标签
plt.title('Performance of clients on data from other clients/GLUE')
plt.xlabel('Clients')  # X轴标签
plt.ylabel('Performance')  # Y轴标签

# 显示图形
plt.show()
plt.close()
fig, ax = plt.subplots()
plt.plot(mean_perf_on_other_clients_bigbench[0], label='cryptonite', color='blue')
plt.plot(mean_perf_on_other_clients_bigbench[1], label='elementary math qa', color='green')
plt.plot(mean_perf_on_other_clients_bigbench[2], label='intersect geometry', color='red')
plt.plot(mean_perf_on_other_clients_bigbench[3], label='list functions', color='purple')
plt.plot(mean_perf_on_other_clients_bigbench[4], label='tracking shuffled objects', color='black')

# 添加图例
plt.legend()

# 设置X轴刻度和标签
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # X轴刻度的值
x_labels = ['Cli-1', 'Cli_2', 'Cli_3', 'Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)

# 添加标题和标签
plt.title('Performance of clients on data from other clients/BigBench')
plt.xlabel('Clients')  # X轴标签
plt.ylabel('Performance')  # Y轴标签

# 显示图形
plt.show()
plt.close()







fig, ax = plt.subplots()
# 绘制折线图
plt.plot(mean_perf_on_other_clients_glue[0], label='mnli/lora', color='blue')
plt.plot(baseline_glue_single[0], label='mnli/single', color='green')
plt.plot(baseline_glue_cv[0], label='mnli/cv', color='red')
plt.plot(baseline_glue_avg[0], label='mnli/avg', color='purple')
# plt.plot(mean_perf_on_other_clients_glue[1], label='qnli', color='green')
# plt.plot(mean_perf_on_other_clients_glue[2], label='qqp', color='red')
# plt.plot(mean_perf_on_other_clients_glue[3], label='sst2', color='purple')

# 添加图例
plt.legend()

# 设置X轴刻度和标签
# x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # X轴刻度的值
x_labels = ['Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)



ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)

# 添加标题和标签
plt.title('Performance of clients on data from other clients/GLUE')
plt.xlabel('Clients')  # X轴标签
plt.ylabel('Performance')  # Y轴标签

# 显示图形
plt.show()







# for i in range(len(mean_perf_on_other_clients_bigbench)):
#     print(mean_perf_on_other_clients_bigbench[i])









# # NAAR, AVG and CV performance on different datasets
# Task = ['MultiNLI', 'Question NLI', 'Stanford Sentiment Treebank', 'Quora Question Pairs', 'Elementary Math QA', 'Cryptonite', 'Intersect Geometry', 'List Functions', 'Tracking Shuffled Objects']
# Single = [85.16, 92.07, 94.41, 86.87, 30.37, 0.76, 24.14, 5.08, 18.85]
# AVG = [84.80, 91.70, 92.77, 85.73, 26.40, 0.27, 15.18, 4.94, 19.04]
# CV = [84.96, 92.15, 93.26, 85.74, 27.47, 0.40, 20.88, 5.01, 21.71]
# NAAR = [85.09, 91.88, 94.25, 81.90, 28.42, 0.35, 18.61, 4.39, 21.33]
#
# import matplotlib.pyplot as plt
#
# # 假设你有四个数组 data1, data2, data3, data4
# Single = [85.16, 92.07, 94.41, 86.87, 30.37, 0.76, 24.14, 5.08, 18.85]
# AVG = [84.80, 91.70, 92.77, 85.73, 26.40, 0.27, 15.18, 4.94, 19.04]
# CV = [84.96, 92.15, 93.26, 85.74, 27.47, 0.40, 20.88, 5.01, 21.71]
# NAAR = [85.09, 91.88, 94.25, 81.90, 28.42, 0.35, 18.61, 4.39, 21.33]
# # 绘制折线图
# plt.plot(Single, label='Single', color='blue')
# plt.plot(AVG, label='AVG', color='green')
# plt.plot(CV, label='CV', color='red')
# plt.plot(NAAR, label='NAAR', color='purple')
# x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # X轴刻度的值
# plt.xticks(x_values, Task)
# # 添加图例
# plt.legend()
#
# # 添加标题和标签
# plt.title('NAAR, AVG and CV performance on different datasets')
# plt.xlabel('Tasks')
# plt.ylabel('Performance')
#
# # 显示图形
# plt.show()

import seaborn as sns
import palettable#python颜色库
import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(8, 6.5))

data = results_on_glue[0][3]
data = np.array(data)
data = data[3:10, 3:10]
# data_p = []
# for i in range(len(data)):
#     temp = []
#     dd = data[i]
#     for j in range(3, 10):
#         temp.append(dd[j])
#     data_p.append(temp)

# data = np.array(data)

sns.heatmap(data=data, cmap='YlGnBu',vmin=82, vmax=87)
plt.ylabel("Model of Client_i")
plt.xlabel("Performance on Dataset_i")

x_labels = ['Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)
plt.title('Performance of Lorahub on Different Data (task on mnli)')
sns.set_theme(style="ticks",font='Times New Roman',font_scale=1)
plt.savefig("LegitimateAmount.svg")
plt.show()
plt.close()

data = results_on_glue[0][5]
data = np.array(data)
data = data[3:10, 3:10]
# data_p = []
# for i in range(len(data)):
#     temp = []
#     dd = data[i]
#     for j in range(3, 10):
#         temp.append(dd[j])
#     data_p.append(temp)

# data = np.array(data)

sns.heatmap(data=data, cmap='YlGnBu',vmin=82, vmax=87)
plt.ylabel("Model of Client_i")
plt.xlabel("Performance on Dataset_i")


x_labels = ['Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)
plt.title('Performance of Single Client on Different Data (task on mnli)')
sns.set_theme(style="ticks",font='Times New Roman',font_scale=1)
plt.savefig("LegitimateAmount.svg")
plt.show()
plt.close()

data = results_on_glue[0][1]
data = np.array(data)
data = data[3:10, 3:10]
# data_p = []
# for i in range(len(data)):
#     temp = []
#     dd = data[i]
#     for j in range(3, 10):
#         temp.append(dd[j])
#     data_p.append(temp)

# data = np.array(data)

sns.heatmap(data=data, cmap='YlGnBu',vmin=82, vmax=87)
plt.ylabel("Model of Client_i")
plt.xlabel("Performance on Dataset_i")


x_labels = ['Cli_4', 'Cli_5', 'Cli_6', 'Cli_7', 'Cli_8', 'Cli_9', 'Cli_10']  # 对应刻度的标签
# plt.xticks(x_values, x_labels)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels)
plt.title('Performance of CV on Different Data (task on mnli)')
sns.set_theme(style="ticks",font='Times New Roman',font_scale=1)
plt.savefig("LegitimateAmount.svg")
plt.show()
plt.close()

