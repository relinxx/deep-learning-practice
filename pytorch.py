import torch

x=torch.tensor([1,2,3])
print(x)
x=torch.zeros(2,3)
print(x)

randn = torch.randn(2,3)
print(randn)

randint = torch.randint(min,max,(2,3))
randint = torch.randint(1,10,(2,3))
print(randint)

a=torch.arange(7,14,3)
print(a)

ras = torch.randn(3,4)
print(ras)

x=torch.arange(1,12)
print(x)
y=x.view(3,4)
print(y)

print(x.view(-1,3))

x=torch.tensor([1,2,3])
usq = x.unsqueeze(1)
print(usq)


uu=torch.randint(1,30,(2,3,4))
print(uu) 