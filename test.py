from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
weights = np.random.uniform(low=-1, high=1, size=10)
weights = torch.tensor(weights).float()


A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
A = torch.tensor(A, dtype=float)
A = F.normalize(A.unsqueeze(0), p=2, dim=1).squeeze(0)



print(A)
