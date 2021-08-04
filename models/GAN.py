import torch
import torch.nn as nn

from lib.normalize import Normalize

__all__ = ['Generator', 'Discriminator']


class Generator(nn.Module):
    """
    Generator for generating hard positive samples.
    A three layer fully connected network.
    Takes (a, p, n) as input and output p'.
    """

    def __init__(self, embedding_size=128):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(3 * embedding_size, 2 * embedding_size)
        self.fc2 = nn.Linear(2 * embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        return self.l2norm(self.fc3(x))


class Discriminator(nn.Module):
    """
    Discriminator for judging triplet (a, p, n) against (a, p', n),
    A three layer fully connected network.
    """

    def __init__(self, embedding_size=128):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(3 * embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size // 2)
        self.fc3 = nn.Linear(embedding_size // 2, 1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        return torch.sigmoid(self.fc3(x).squeeze(-1))
