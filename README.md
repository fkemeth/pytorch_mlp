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

    python main.py


ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repo](https://github.com/fkemeth/pytorch_mlp).




LICENCE
---------


This work is licensed under MIT license.
