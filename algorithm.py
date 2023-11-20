from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from lorahub import lorahub_learning
from itertools import combinations
from utils import evaluation
from copy import deepcopy
import numpy as np
from tqdm import tqdm

def average_aggregation(base_model, lora_adaptors):
    weight = 1/len(lora_adaptors)
    final_state_dict = {}
    keys = lora_adaptors[0].keys()
    for i, lora_adaptor in enumerate(lora_adaptors):
        if i == 0:
            for key in keys:
                final_state_dict[key] = weight * lora_adaptor[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weight * lora_adaptor[key]
                )
    return update_lora_weights(base_model, final_state_dict)

def lorahub_aggregation(model, lora_adaptors, data, tokenizer, batch_size, sample_size = 5, max_inference_step = 40, seed = 42):
    data = data[:sample_size]
    weights, model = lorahub_learning(lora_adaptors = lora_adaptors,
                                       model = model,
                                       data = data,
                                       tokenizer = tokenizer,
                                       batch_size = batch_size,
                                       max_inference_step = max_inference_step,
                                       seed = seed)
    return weights, model

def load_lora_adaptors(args):
    lora_adaptors = []
    for i in range(args.num_clients):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, return_dict=True)
        lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{args.task}-{i}')
        lora_adaptors.append(get_peft_model_state_dict(lora_model))
    return lora_adaptors

def update_lora_weights(model, state_dict):
    set_peft_model_state_dict(model,state_dict)
    model = model.merge_and_unload()
    return model

def combination(n, k):
    # generate C_n_k, n is the total number of adapters, k is the number of error clients
    elements = range(n)
    combinations_n_k = list(combinations(elements, k))
    return combinations_n_k

def cross_validation(base_model, lora_adaptors, num_error_client, data, tokenizer, max_search_epoch=30, sampling = 50):
    # "adapters" should be a list of adapters
    total = len(lora_adaptors)
    combination_n_k = combination(total, num_error_client)

    aggregated = []
    selected = []
    choice = np.random.choice(len(combination_n_k), max_search_epoch)
    combination_n_k = [combination_n_k[i] for i in choice]
    for outside in tqdm(combination_n_k):
        inside = []
        select = []
        for i, lora_adaptor in enumerate(lora_adaptors):
            if i not in outside:
                # aggregate
                select.append(i)
                inside.append(lora_adaptor)
        selected.append(select)
        weights = average_aggregation(deepcopy(base_model), inside)
        aggregated.append(weights)
    performance = -1
    best = -1
    for i in range(len(aggregated)):
        model = aggregated[i]
        task_perf_cross_val, example_predictions_error = evaluation(data, model, tokenizer, batch_size=32, sampling = sampling)
        if performance < task_perf_cross_val:
            performance = task_perf_cross_val
            best = i
    return aggregated[best], selected[best]

