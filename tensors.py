import torch
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# creating tensors

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True).to(device)
x = torch.empty(2, 3, dtype=torch.float32)
x = torch.rand(2, 3)
x = torch.eye(2, 3)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0, end=5, steps=5)
x = torch.empty((1, 5)).normal_(mean=0, std=1)
x = torch.diag(torch.tensor([1, 2, 3, 4, 5]))

# converting tensors

my_tensor = torch.arange(4)
x = my_tensor.bool()
x = my_tensor.short()
x = my_tensor.int()
x = my_tensor.long()
x = my_tensor.float() # float32
x = my_tensor.double() # float64
x = my_tensor.half() # float16
x = my_tensor.bfloat16() # bfloat16


arr = np.ones((5, 5))
x = torch.from_numpy(arr)
arr = x.numpy()


# tensor math

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z = x + y
z = x - y
z = x * y
z = x / y
z = x % y
z = x ** 2
z = x ** y
z = torch.dot(x, y)

# martrix multiplication

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))

z = torch.mm(x1, x2)
z = x1 @ x2


# aggregations

x = torch.tensor([1, 2, 3])
z = torch.sum(x, dim=0)
z = torch.prod(x)
z = torch.mean(x.float()) # requires float
z = torch.abs(x)
z = torch.min(x, dim=0)
z = torch.max(x, dim=0)
z = torch.argmax(x, dim=0)
z = torch.eq(x, x)
z = torch.sort(x, dim=0, descending=True).values
z = torch.clamp(x, min=5)
z = torch.any(x <= 0)


# indexing

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z = x[1, 1]
z = x[1, :]
z = x[:, 2]
z = x[1:3, 1:3]
z = x[1:3, :]
z = x[(x < 2) | (x > 5)]
z = torch.where(x > 5, x, x * 2)


# reshaping

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.view(3, 2)
z = x.reshape(3, 2)
z = x.unsqueeze(dim = 1)

print(z)

