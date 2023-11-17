from datasets import load_dataset
import random
import numpy as np
import torch

def k_split(num_clients,num_error_clients,dataset):
    data = []
    for i in range(num_clients):
        subdata = dataset.shard(num_clients,i)
        if i < num_error_clients:
            random.shuffle(subdata['target'])
        data.append(subdata)
    return data

def split_data(args):
    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/NLP524", args.task).shuffle(seed=args.seed)
    train_ds = k_split(args.num_clients,args.num_error_clients,dataset['train'])
    test_ds = k_split(args.num_clients,args.num_error_clients,dataset['test'])
    valid_ds = k_split(args.num_clients,args.num_error_clients,dataset['valid'])
    return (train_ds, test_ds, valid_ds)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)