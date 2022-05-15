"""
Copyright © 2022 Felix P. Kemeth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import configparser

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from utils import DenseStack, Model, progress


def main(config: configparser.ConfigParser):
    """
    Train a multilayer perceptron on MNIST dataset.

    Args:
        config: Config parser containing hyperparameters
    """
    # transformations = [transforms.ToTensor(), transforms.Lambda(
    #     lambda x: torch.flatten(x))]
    transformations = [transforms.ToTensor(), transforms.Lambda(torch.flatten)]

    dataset_train = torchvision.datasets.MNIST(config['Data']['path'], train=True, download=True,
                                               transform=transforms.Compose(transformations))
    dataset_test = torchvision.datasets.MNIST(config['Data']['path'], train=False, download=True,
                                              transform=transforms.Compose(transformations))

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config['Training'].getint('batch_size'), shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['Training'].getint('batch_size'), shuffle=True)

    network = DenseStack(cfg['Model'])

    model = Model(dataloader_train, dataloader_test,
                  network, config['Training'])

    progress_bar = tqdm(
        range(0, config['Training'].getint('epochs')), desc=progress(0, 0))

    for _ in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        progress_bar.set_description(progress(train_loss, val_loss))


if __name__ == '__main__':
    cfg = configparser.ConfigParser()
    cfg.read('config.cfg')
    main(cfg)
