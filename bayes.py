import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import transforms, datasets

# переводит изображение в оттенки серого, в тензор, и выполняет нормализацию
data_transform = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(),
     transforms.Normalize(mean=[0], std=[1])])

# download=True если запускается в первый раз (датасет не скачан)
mnist_train = datasets.MNIST(root='./data', train=True, download=False,
                             transform=data_transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=False,
                            transform=data_transform)

# ОБУЧЕНИЕ

xcount = torch.ones((784, 10), dtype=torch.float32)
ycount = torch.zeros(10, dtype=torch.float32)

# data - насколько сильно пиксель горит (float от 0 до 1)
# label - какому классу принадлежит
ycount += 2
for data, label in mnist_train:
    data = torch.round(data).reshape(784)
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += data.reshape(784)
ycount -= 2


# изображение принадлежит одному из 10 классов
py = ycount / sum(ycount)
# пиксель горит, при условии, что он принадлежит классу y
pxy = xcount / ycount.reshape(1, 10)


# визуализация тепловых портретов цифр
def heatmap(input_pxy):
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for i in range(10):
        pxy = input_pxy[:, i].reshape(28, 28)
        sns.heatmap(pxy, cmap="magma", center=0, ax=axes[i], square=True,
                    xticklabels=False, yticklabels=False, cbar=False)
    plt.show()


heatmap(pxy)

# softmax
for data, label in mnist_test:
    data = torch.round(data).reshape(784)
    pred = torch.zeros(10)
    for i in range(10):
        evaluation = torch.log(py[i]) + torch.sum(torch.log(data * pxy[:, i] + (1-data) * (1-pxy[:, i])))
        pred[i] = evaluation
    pred -= torch.max(pred) # magic
    pred = torch.exp(pred)
    pred /= torch.sum(pred)
    print(pred, label)

