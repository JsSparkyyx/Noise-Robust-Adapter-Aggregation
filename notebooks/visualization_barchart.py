import os
import pandas as pd
import matplotlib.pyplot as plt

bigbench_path = "results/bigbench/"
glue_path = "results/glue/"

tasks_bigbench = ['elementary_math_qa','cryptonite','intersect_geometry','list_functions', 'tracking_shuffled_objects']
tasks_glue = ['mnli','qnli','sst2','qqp']
import numpy as np



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

mean_perf_on_other_clients_glue = []
for task in results_on_glue:
    lora_perf = task[3]
    client_perf = []
    sum = 0
    for j in range(3, 10):
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += lora_perf[j][k]
    sum /= 49
    mean_perf_on_other_clients_glue.append(sum)

baseline_glue_single = []
for task in results_on_glue:
    single_perf = task[5]
    client_perf = []
    sum = 0
    for j in range(3, 10):
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += single_perf[j][k]
    sum /= 49
    baseline_glue_single.append(sum)


baseline_glue_cv = []
for task in results_on_glue:
    cv_perf = task[1]
    client_perf = []
    sum = 0
    for j in range(3, 10):
        for k in range(3, 10):
            # if k == j:
            #     continue
            sum += cv_perf[j][k]
    sum /= 49
    baseline_glue_cv.append(sum)

baseline_glue_avg = []
for task in results_on_glue:
    avg_perf = task[0][3:-2]
    baseline_glue_avg.append(avg_perf)

print(mean_perf_on_other_clients_glue)
print(single_perf)
print(cv_perf)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

max_lst_of_all = {}
max_lst_of_all[0] = [mean_perf_on_other_clients_glue[0], baseline_glue_single[0], baseline_glue_cv[0]]
max_lst_of_all[1] = [mean_perf_on_other_clients_glue[1], baseline_glue_single[1], baseline_glue_cv[1]]
max_lst_of_all[2] = [mean_perf_on_other_clients_glue[2], baseline_glue_single[2], baseline_glue_cv[2]]
max_lst_of_all[3] = [mean_perf_on_other_clients_glue[3], baseline_glue_single[3], baseline_glue_cv[3]]

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax1) for the outliers, and the bottom
# (ax2) for the details of the majority of our data


plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rc('font',family='Times New Roman')
plt.rc('font',size=14)
width=0.2
plt.bar([i for i in range(len(mean_perf_on_other_clients_glue))], mean_perf_on_other_clients_glue, width=width, label='LoRAHub')
plt.bar([i+width for i in range(len(baseline_glue_single))], baseline_glue_single, width=width, label='Single', )
plt.bar([i+width*2 for i in range(len(baseline_glue_cv))], baseline_glue_cv, width=width, label='CV', color='red', )
plt.xticks([x+width/2 for x in range(4)], ['MNLI', 'QNLI', 'QQP', 'SST2'])
plt.title('Performance on Data From Other Clients (Tasks on GLUE)')
plt.ylim(80, 95)
plt.xlabel("Tasks")
plt.ylabel("Accuracy/%")
plt.legend(loc='upper left')
plt.show()