import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from algorithm import *

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

# weights = np.random.uniform(low=-1, high=1, size=10)

# class Model(nn.Module):
#     def __init__(self, num_of_modules):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(num_of_modules, num_of_modules)
#
#     def forward(self, x):
#         return torch.softmax(self.fc(x), dim=1)

# class Model(nn.Module):
#     def __init__(self, weight):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(in_features=10, out_features=1)
#         self.weight = weight
#
#     def forward(self, x):
#         return torch.sigmoid(self.fc(x) * self.weight)


weights = results_on_glue[0][4]

# linear_layer = nn.Linear(in_features=1, out_features=10)

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.weights = nn.Parameter(torch.rand(1, 10))
#
#     def forward(self, x):
#         self.weights = nn.functional.normalize(self.weights, p=2, dim=1)
#         return torch.matmul(x, self.weights.view(-1, 1))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_layer = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        # 在前向传播中使用线性层
        output = self.linear_layer(x)
        return output


model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# for i in range(num_clients):
#     base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
#     lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-{i}')
#     lora_adaptors.append(get_peft_model_state_dict(lora_model))



# model = Model(len(weights))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# weights = torch.tensor(weights)
input = np.array(results_on_glue[0][3])
input = input[:-1, :-1]
input = torch.tensor(input).to(torch.float32)
training_steps = 200
for i in tqdm(range(training_steps)):

    # aggregated_weights = model(weights)
    output = model(input)
    model.linear_layer.weight.data = nn.functional.softmax(model.linear_layer.weight.data, dim=1)
    loss = 100 - (torch.sum(output) / len(input))
    ########
    # 合并成一个lora
    # loss = evaludation


    ########
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss.item())
    # model.linear_layer.weight.data = nn.functional.normalize(model.linear_layer.weight.data, p=2, dim=1)
    # model.linear_layer.weight.data = nn.functional.relu(model.linear_layer.weight.data)



