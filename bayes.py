import torch
from torchvision import transforms, datasets

# переводит изображение в оттенки серого, в тензор и выполняет нормализацию
data_transform = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(),
     transforms.Normalize(mean=[0], std=[1])])

# download=True если запускается в первый раз (датасет не скачен)
mnist_train = datasets.MNIST(root='./data', train=True, download=False,
                             transform=data_transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=False,
                            transform=data_transform)

xcount = torch.ones((784, 10), dtype=torch.float32)
ycount = torch.ones(10, dtype=torch.float32)

for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += data.reshape(784)

print(ycount / sum(ycount))