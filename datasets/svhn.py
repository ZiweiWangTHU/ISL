from __future__ import print_function
from PIL import Image
import numpy as np
import torchvision.datasets as datasets


class SVHNInstance(datasets.SVHN):
    """SVHNInstance Dataset.
    """

    def __init__(self, **kwargs):
        super(SVHNInstance, self).__init__(**kwargs)
        self.train_labels = self.labels

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
