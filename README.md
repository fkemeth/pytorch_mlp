OVERVIEW
---------

This repository provides function to easily create multilayer perceptrons in pytorch for either
classification or regression tasks. Each layer of the multilayer perceptron is composed of

- dense linear layer,
- batch normalization (optional),
- dropout (optional),
- GELU element-wise activation (except after last layer).

The repository also contains a Model wrapper class with easy to use train and helper functions,
the usage of which is illustrated on the MNIST dataset.

INSTALLATION
---------

Via source

    git clone https://github.com/fkemeth/pytorch_mlp
    cd pytorch_mlp/
    pip install -r requirements.txt

EXAMPLE USAGE
---------

As an illustrative example, we use the MNIST dataset containing images with digits from 0 to 9.

    import torch
    import torchvision
    import torchvision.transforms as transforms

    from tqdm.auto import tqdm

    from utils import DenseStack, Model, progress

    transformations = [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]

    dataset_train = torchvision.datasets.MNIST('data/', train=True, download=True,
                                               transform=transforms.Compose(transformations))
    dataset_test = torchvision.datasets.MNIST('data/', train=False, download=True,
                                              transform=transforms.Compose(transformations))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=True)

    network = DenseStack(int(28*28), 10, [64, 64, 64], use_batch_norm=True, dropout_rate=0.5)

    model = Model(dataloader_train, dataloader_test, network, classification=True)

    progress_bar = tqdm(range(0, 40), desc=progress(0, 0))

    for _ in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        progress_bar.set_description(progress(train_loss, val_loss))



ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repo](https://github.com/fkemeth/pytorch_mlp).




LICENCE
---------


This work is licensed under MIT license.
