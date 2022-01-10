import matplotlib.pyplot as plt
import numpy as np

x_labels = ['SVHN (clf.)', 'MNIST (clf.)', 'MNIST (clt.)', 'MNIST (tnf.)']
models = ['baseline', 'classifier', 'encoder', 'discriminator']

data = [
    [83.15, 99.02, 60.28, 56.05],
    [91.06, 99.44, 0, 71.36],
    [82.82, 99.28, 78.45, 52.17],
    [89.91, 99.41, 83.35, 72.39]
]


def main():
    x = np.arange(4)
    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.ylabel('Accuracy')
    plt.bar(x, data[0], color="#4EC6FF", width=width, label=models[0])
    plt.bar(x + width, data[1], color="#FFBE15", width=width, label=models[1])
    plt.bar(x + 2 * width, data[2], color="#56FF56", width=width, label=models[2])
    plt.bar(x + 3 * width, data[3], color="#FF7578", width=width, label=models[3])
    plt.xticks([0, 1, 2, 3], x_labels)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
