import torch


a = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
b = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
c = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
print(a.shape)
print(torch.cat([a, b, c], dim=1).shape)