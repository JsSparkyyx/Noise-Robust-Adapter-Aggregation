from datasets import load_dataset

def get_data(name,task):
    data = load_dataset(name,task)
    
    return