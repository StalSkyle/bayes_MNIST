from random import randint

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

xcount = torch.zeros((784, 10), dtype=torch.float32)
ycount = torch.zeros(10, dtype=torch.float32)

# data - горит пиксель или нет
# label - какому классу принадлежит
for data, label in mnist_train:
    data = torch.round(data).reshape(784)  # пиксель либо горит, либо нет
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += data.reshape(784)

# изображение принадлежит одному из 10 классов
py = ycount / sum(ycount)
# вероятность того, что пиксель горит, при условии, что он принадлежит классу y
pxy = xcount / ycount.reshape(1, 10)

# визуализация тепловых портретов цифр
input_pxy = torch.clone(pxy)
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
fig.suptitle("Тепловые портреты цифр",
             y=0.9, fontsize=30, fontweight='bold')
for i in range(10):
    pxy_ = input_pxy[:, i].reshape(28, 28)
    sns.heatmap(pxy_, cmap="magma", center=0, ax=axes[i], square=True,
                xticklabels=False, yticklabels=False, cbar=False)
plt.tight_layout()
plt.show()


# построение предсказания, используется softmax
def calculate_pred(data):
    data = torch.round(data).reshape(784)
    pred = torch.zeros(10)
    for i in range(10):
        evaluation = torch.log(py[i]) + torch.sum(
            torch.log(data * pxy[:, i] + (1 - data) * (1 - pxy[:, i])))
        pred[i] = evaluation
    pred -= torch.max(pred)  # избежание переполнения
    pred = torch.exp(pred)
    pred /= torch.sum(pred)
    return pred


# расчет точности модели
quality = []
for data, label in mnist_test:
    pred = calculate_pred(data)
    quality.append(bool(torch.argmax(pred) == label))

print("Точность модели:", sum(quality) / len(quality))

# визуализация нескольких предсказаний
rnd = [randint(0, 10000) for _ in range(10)]
fig, axes = plt.subplots(2, 10, figsize=(18, 6))
fig.suptitle("Примеры распознавания цифр",
             fontsize=30, fontweight='bold')
for i in range(10):
    data, num = mnist_train[rnd[i]]
    data = data.reshape(28, 28)
    sns.heatmap(data, cmap="magma", center=0, ax=axes[0, i], square=True,
                xticklabels=False, yticklabels=False, cbar=False)
    pred = list(calculate_pred(data))
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    axes[1, i].bar(xs, pred)
plt.tight_layout()
plt.show()
