import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['glue'], default='glue')
    parser.add_argument('--task', type=str, choices=['mnli','qnli','sst2','qqp'], default='mnli')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_error_clients', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--model', type=str, choices=['google/flan-t5-base'], default='google/flan-t5-base')

    args = parser.parse_args()
    return args