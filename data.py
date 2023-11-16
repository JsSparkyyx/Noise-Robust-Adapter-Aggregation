from datasets import load_dataset
import random

def k_split(num_clients,num_error,dataset):
    data = []
    for i in range(num_clients):
        subdata = dataset.shard(num_clients,i)
        if i < num_error:
            random.shuffle(subdata['target'])
        data.append(subdata)
    return data

def split_data(args):
    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/processed_glue", args.task).shuffle(seed=args.seed)
    train_ds = k_split(args.num_clients,args.num_error,dataset['train'])
    test_ds = k_split(args.num_clients,args.num_error,dataset['test'])
    valid_ds = k_split(args.num_clients,args.num_error,dataset['valid'])
    return (train_ds, test_ds, valid_ds)