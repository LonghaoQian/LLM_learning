import torch
print(torch.__version__)
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.tensor([[1.0, -0.2], [2.0, -3.0]],
                 dtype=torch.float32, 
                 device=device)
print(a.shape)

a_mean_0 = a.mean(dim=0, keepdim=True)
print(a_mean_0)
print(a_mean_0.shape)
a_mean_1 = a.mean(dim=1, keepdim=True)
print(a_mean_1)
print(a_mean_1.shape)