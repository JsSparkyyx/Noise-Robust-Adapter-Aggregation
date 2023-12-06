# the code is implemented from https://github.com/sail-sg/lorahub
from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig
from functools import partial
from typing import List, Optional, Union
import copy
import numpy as np

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
    return peft_model, tokenizer, cache

def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    # use gpu if available
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    logits_list = []
    decoder_hidden_states_list = []
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.decoder_hidden_states
            embedding = hidden_states[0]
            outputs_ = hidden_states[1]
            logits_list.append(logits)
            decoder_hidden_states_list.append(outputs_)
            train_loss += loss.detach().float()
    loss = train_loss.float()
    logits = logits_list[0]
    for i in range(1, len(logits_list)):
        logits += logits_list[i]
    decoder_hidden_states = decoder_hidden_states_list[0]
    for i in range(1, len(decoder_hidden_states_list)):
        decoder_hidden_states += decoder_hidden_states_list[i]
    # average loss over the number of examples
    return float(loss) / len(example_dataset["input"]), logits, decoder_hidden_states

def default_kl_rep(p, q):
    kl_div = 0.0
    for i in range(len(p)):
        kl_div += torch.sum(p[i] * torch.log(p[i] / q[i]))
    return kl_div.item()


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def default_l2_regularization(weights):
    """
    Get the L2 regularization term for the weights
    """
    sum_of_squares = sum([(x * x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular, get_kl_rep):
    # the composed lora state dict
    single = model.state_dict()
    single_model = []
    for key in single.keys():
        single_model.append(single[key])
    _, logits_single, hidden_states_single = get_loss(example_dataset, model, batch_size)
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
    aggregated = model.state_dict()
    aggregated_model = []
    for key in aggregated.keys():
        aggregated_model.append(aggregated[key])
    # minimize the metric
    loss, logits_aggregated, hidden_states_aggregated = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    logits_single = torch.mean(logits_single, dim=1)
    logits_aggregated = torch.mean(logits_aggregated, dim=1)
    hidden_states_single = torch.mean(hidden_states_single, dim=1)
    hidden_states_aggregated = torch.mean(hidden_states_aggregated, dim=1)
    # L2 = default_l2_regularization(weights)
    # KL = get_kl_rep(single_model, aggregated_model)
    L2_logits = torch.norm(logits_single - logits_aggregated, p=2).item()
    KL_logits = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits_single, dim=1), torch.nn.functional.softmax(logits_aggregated, dim=1), reduction='batchmean').item()
    L2_hidden_states = torch.norm(hidden_states_single - hidden_states_aggregated, p=2).item()
    KL_hidden_states = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(hidden_states_single, dim=1),
                                           torch.nn.functional.softmax(hidden_states_aggregated, dim=1),
                                           reduction='batchmean').item()
    print(L2_hidden_states)
    print(KL_hidden_states)
    metric_val = loss + get_regular(weights) + KL_hidden_states
    
    return metric_val

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict
    
def lorahub_inference(example_inputs: List[str],
                      model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
                      tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
                      batch_size: int,
                      # if not provided, we do not report the accuracy
                      example_outputs: List[str]=None):
    
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        return correct / total * 100

    example_predictions = []
    # load model
    if isinstance(model_or_name_path, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
    else:
        model = model_or_name_path
    
    # load tokenizer
    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
    else:
        tokenizer = tokenizer_or_tokenizer_path
            
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256
        )
        outputs = tokenizer.batch_decode(
            outputs.to("cpu"), skip_special_tokens=True
        )
        example_predictions.extend(outputs)
    
    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None
        
    return example_predictions, task_perf


def adlorahub_learning(lora_adaptors, 
                     model,
                     data,
                     tokenizer,
                     max_inference_step: int,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     get_kl_rep=default_kl_rep,
                     seed=42):
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    number_of_loras = len(lora_adaptors)
    cache = {}
    lora_module_list = [i for i in range(number_of_loras)]
    cache = {i:lora_adaptors[i] for i in lora_module_list}
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    example_inputs, example_outputs = data['source'], data['target']
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    get_score_partial = partial(get_score, 
                                model=model, 
                                cache=cache,
                                example_dataset=dataset,
                                batch_size=batch_size,
                                get_loss=get_loss, 
                                get_regular=get_regular,
                                get_kl_rep=get_kl_rep)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model