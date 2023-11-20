from datasets import load_dataset, Dataset, DatasetDict
import random
import numpy as np
import torch
from tqdm import trange

def k_split(num_clients,num_error_clients,dataset, data_name):
    data = []
    for i in range(num_clients):
        subdata = dataset.shard(num_clients,i)
        target = subdata['target']
        source = subdata['source']
        if i < num_error_clients:
            random.shuffle(target)
        if data_name == "bigbench":
            target = [i[0] for i in target]
        subdata = Dataset.from_dict({'source':source, 'target':target})
        data.append(subdata)
    return data

def split_data(args):
    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/NLP524", args.task).shuffle(seed=args.seed)
    if args.dataset == 'bigbench':
        dataset = load_dataset("tasksource/bigbench", args.task).shuffle(seed=args.seed)
        dataset = dataset.rename_columns({'inputs':'source','targets':'target'})
    train_ds = k_split(args.num_clients,args.num_error_clients,dataset['train'],args.dataset)
    if args.dataset == 'glue':
        valid_ds = k_split(args.num_clients,args.num_error_clients,dataset['valid'],args.dataset)
    else:
        valid_ds = k_split(args.num_clients,args.num_error_clients,dataset['validation'],args.dataset)
    return (train_ds, valid_ds)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def retrive_data(ds,number):
    '''
    retrive the data of client i
    '''
    (train_ds, valid_ds) = ds
    return DatasetDict({'train':train_ds[number],'valid':valid_ds[number]})

def accuracy_score(outputs, ground_truths):
    correct = 0
    total = 0
    for output, truth in zip(outputs, ground_truths):
        if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
            correct += 1
        total += 1
    return correct / total * 100

def evaluation(data, model, tokenizer, batch_size = 128):
    example_predictions = []
    eval_set = "valid"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in trange(0, len(data[eval_set]["source"]), batch_size):
            inputs = tokenizer(
                    data[eval_set]["source"][i : i + batch_size],
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

    task_perf = accuracy_score(example_predictions, data[eval_set]["target"])
    return task_perf, example_predictions