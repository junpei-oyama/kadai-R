from typing import List

from matplotlib import pyplot as plt


def plot_accuracy(accuracy_list: List[float]) -> None:
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def plot_loss(loss_list: List[float]) -> None:
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def main():
    accuracy_list = list(range(0, 10, 1))
    plot_accuracy(accuracy_list)


if __name__ == '__main__':
    main()
