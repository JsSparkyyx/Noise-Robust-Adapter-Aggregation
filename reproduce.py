from utils import evaluation, set_seed, k_split, retrive_data
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, get_peft_model_state_dict
from algorithm import lorahub_aggregation, average_aggregation, cross_validation
import numpy as np
from tqdm import tqdm
import os

task_set = ['mnli','qnli','sst2','qqp','elementary_math_qa','cryptonite','intersect_geometry','list_functions','tracking_shuffled_objects']
# task_set = ['elementary_math_qa']
seed = 42
num_clients = 10
num_error_clients = 3
lora_sample_size = 5
result_dir = './results'
model_name_or_path = 'google/flan-t5-base'
eval_batch_size = 64
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
set_seed(seed)

def split_data(data_name, task):
    if data_name == 'glue':
        dataset = load_dataset("JsSparkYyx/NLP524", task).shuffle(seed=seed)
    if data_name == 'bigbench':
        dataset = load_dataset("tasksource/bigbench", task).shuffle(seed=seed)
        dataset = dataset.rename_columns({'inputs':'source','targets':'target'})
    train_ds = k_split(num_clients,num_error_clients,dataset['train'],data_name)
    if data_name == 'glue':
        valid_ds = k_split(num_clients,num_error_clients,dataset['valid'],data_name)
    else:
        valid_ds = k_split(num_clients,num_error_clients,dataset['validation'],data_name)
    return (train_ds, valid_ds)

def reproduce_average_aggregation():
    print('Evaluation for method average aggregation')
    for task in tqdm(task_set):
        results = np.zeros(num_clients+1)
        data_name = 'glue' if task in ['mnli','qnli','sst2','qqp'] else 'bigbench'
        task_dir = os.path.join(result_dir, data_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        lora_adaptors = []
        for i in range(num_clients):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-{i}')
            lora_adaptors.append(get_peft_model_state_dict(lora_model))
        ds = split_data(data_name, task)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
        base_lora = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-0')
        avg_model = average_aggregation(base_lora,lora_adaptors)
        for number in range(num_clients):
            data = retrive_data(ds, number)
            task_perf, example_predictions = evaluation(data, avg_model, tokenizer, data_name, batch_size=eval_batch_size)
            results[number] = task_perf
        results[num_clients] = np.mean(results[:num_clients])
        print(results)
        np.savetxt(os.path.join(task_dir,f'{task}_avg.csv'), results, delimiter=',')

def reproduce_single_model():
    print('Evaluation for method single model')
    for task in tqdm(task_set):
        results = np.zeros((num_clients+1,num_clients+1))
        data_name = 'glue' if task in ['mnli','qnli','sst2','qqp'] else 'bigbench'
        task_dir = os.path.join(result_dir, data_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        ds = split_data(data_name, task)
        for number in range(num_clients):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            base_lora = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-{number}')
            for i in range(num_clients):
                data = retrive_data(ds, i)
                task_perf, example_predictions = evaluation(data,base_lora,tokenizer, batch_size=eval_batch_size)
                results[number,i] = task_perf
            results[number,num_clients] = np.mean(results[number,:num_clients])
        results[num_clients] = np.mean(results[:num_clients],axis=0)
        print(results)
        np.savetxt(os.path.join(task_dir,f'{task}_single.csv'), results, delimiter=',')

def reproduce_lorahub_aggregation():
    print('Evaluation for method lorahub aggregation')
    for task in tqdm(task_set):
        results = np.zeros((num_clients+1,num_clients+1))
        lora_weights = np.zeros((num_clients,num_clients))
        data_name = 'glue' if task in ['mnli','qnli','sst2','qqp'] else 'bigbench'
        task_dir = os.path.join(result_dir, data_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        lora_adaptors = []
        for i in range(num_clients):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-{i}')
            lora_adaptors.append(get_peft_model_state_dict(lora_model))
        ds = split_data(data_name, task)
        for number in range(num_clients):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            base_lora = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-0')
            data = retrive_data(ds, number)
            weights, lorahub_model = lorahub_aggregation(base_lora, lora_adaptors, data["valid"], tokenizer, batch_size = lora_sample_size, sample_size = lora_sample_size, seed = seed)   
            for i in range(num_clients):
                data = retrive_data(ds, i)
                task_perf, example_predictions = evaluation(data,lorahub_model,tokenizer, batch_size=eval_batch_size)
                results[number,i] = task_perf
            lora_weights[number] = weights
            results[number,num_clients] = np.mean(results[number,:num_clients])
        results[num_clients] = np.mean(results[:num_clients],axis=0)
        print(results)
        print(lora_weights)
        np.savetxt(os.path.join(task_dir,f'{task}_lorahub.csv'), results, delimiter=',')
        np.savetxt(os.path.join(task_dir,f'{task}_lorahub_weights.csv'), lora_weights, delimiter=',')

def reproduce_cross_validation_aggregation():
    print('Evaluation for method cross validation aggregation')
    for task in tqdm(task_set):
        results = np.zeros((num_clients+1,num_clients+1))
        cv_weights = np.zeros((num_clients,num_clients-num_error_clients))
        data_name = 'glue' if task in ['mnli','qnli','sst2','qqp'] else 'bigbench'
        task_dir = os.path.join(result_dir, data_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        lora_adaptors = []
        for i in range(num_clients):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-{i}')
            lora_adaptors.append(get_peft_model_state_dict(lora_model))
        ds = split_data(data_name, task)
        for number in range(3,num_clients):
            data = retrive_data(ds, number)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            base_lora = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{task}-0')
            cv_model, weights = cross_validation(base_lora, lora_adaptors, num_error_clients, data, tokenizer)
            for i in range(3,num_clients):
                data = retrive_data(ds, i)
                task_perf, example_predictions = evaluation(data,cv_model,tokenizer, batch_size=eval_batch_size)
                results[number,i] = task_perf
            cv_weights[number] = weights
            results[number,num_clients] = np.mean(results[number,3:num_clients])
        results[num_clients,3:] = np.mean(results[:num_clients,3:],axis=0)
        print(results)
        np.savetxt(os.path.join(task_dir,f'{task}_cv.csv'), results, delimiter=',')
        np.savetxt(os.path.join(task_dir,f'{task}_cv_weights.csv'), cv_weights, delimiter=',')

if __name__ == '__main__':
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # reproduce_average_aggregation()
    # reproduce_single_model()
    # reproduce_lorahub_aggregation()
    reproduce_cross_validation_aggregation()