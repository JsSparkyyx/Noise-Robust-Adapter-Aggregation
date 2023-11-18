from datasets import load_dataset, Dataset
import random
import numpy as np
import torch

def k_split(num_clients,num_error_clients,dataset):
    data = []
    for i in range(num_clients):
        subdata = dataset.shard(num_clients,i)
        if i < num_error_clients:
            target = subdata['target']
            random.shuffle(target)
            source = subdata['source']
            subdata = Dataset.from_dict({'source':source, 'target':target})
        data.append(subdata)
    return data

def split_data(args):
    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/NLP524", args.task).shuffle(seed=args.seed)
    if args.dataset == 'bigbench':
        dataset = load_dataset("tasksource/bigbench", args.task).shuffle(seed=args.seed)
        dataset = dataset.rename_columns({'inputs':'source','targets':'target'})
    train_ds = k_split(args.num_clients,args.num_error_clients,dataset['train'])
    if args.dataset == 'glue':
        test_ds = k_split(args.num_clients,args.num_error_clients,dataset['test'])
        valid_ds = k_split(args.num_clients,args.num_error_clients,dataset['valid'])
    else:
        test_ds = None
        valid_ds = k_split(args.num_clients,args.num_error_clients,dataset['validation'])
    return (train_ds, test_ds, valid_ds)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)