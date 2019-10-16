import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].reshape(1, 28, 28).astype(np.float32), self.y[idx]


class TestDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].reshape(1, 28, 28).astype(np.float32)


def main():
    X_dummy = np.ones(shape=(5, 784))
    y_dummy = np.ones(shape=(5,))

    train_data = TrainDataset(X_dummy, y_dummy)

    image, label = train_data[0]
    print(image.shape)
    print(label)

    train_loader = DataLoader(train_data, batch_size=2)

    for images, labels in train_loader:
        print(images.size(), labels.size())


if __name__ == '__main__':
    main()
