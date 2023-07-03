import torch

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
print(torch.outer(t1, t2).T)
print(torch.mm(t1.unsqueeze(1), t2.unsqueeze(0)))