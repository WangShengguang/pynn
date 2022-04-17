import torch

# x = Variable(torch.tensor([0,1,2,3], dtype=torch.float), requires_grad=True)
# y = Variable(torch.tensor([1,2,3,4], dtype=torch.float), requires_grad=True)
# z = x*y
#
# print(z)
# y.backward()
# print(z.grad)
# print(x.grad)
# print(y.grad)
# -------------
# x = torch.tensor([1.0, 3.0], requires_grad=True)
# # y = torch.tensor([0.0, 2.0], requires_grad=True)
# y = torch.pow(x, 2)
# # y.backward()
# print(x.grad)
# print(y.grad)
#
# z = (x * y).sum()
# z.backward(retain_graph=True)  # 函数的求导方法
# print(x.grad)
# print(y.grad)

#------------------------
# from torch.autograd import Variable
# x = Variable(torch.randn(2, 2), requires_grad=True)
# y = Variable(torch.randn(2, 2))
# b = x * y
#
# print(x.requires_grad)
# print(y.requires_grad)
#
# print(b.requires_grad)

#-------
x = torch.tensor([0.0, 2.0, 8.0], requires_grad=True)
y = torch.tensor([5.0, 1.0, 7.0], requires_grad=True)
z = x * y
# z.backward(torch.FloatTensor([1.0, 1.0, 1.0]))
z = z.sum()
z.backward()
print(x.grad)

torch.nn.Embedding