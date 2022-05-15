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
from ast import literal_eval
from configparser import SectionProxy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def progress(train_loss: float, val_loss: float) -> str:
    """
    Create progress bar description.

    Args:
        train_loss: Training loss
        val_loss: Validation or test loss

    Returns:
        String with training and test loss
    """
    return 'Train/Loss: {:.8f} ' \
           'Val/Loss: {:.8f}' \
           .format(train_loss, val_loss)


class DenseStack(nn.Module):
    """
    Fully connected neural network.

    Args:
        config: Configparser section proxy with:
            num_in_features: Number of input features
            num_out_features: Number of output features
            num_hidden_features: List of nodes in each hidden layer
            use_batch_norm: If to use batch norm
            dropout_rate: If, and with which rate, to use dropout
    """

    def __init__(self, config: SectionProxy) -> None:
        super().__init__()
        self.use_batch_norm = config.getboolean('use_batch_norm')
        self.dropout_rate = config.getfloat('dropout_rate')

        self.fc_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        self.acts = []

        in_features = config.getint('input_size')
        # List containing number of hidden and output neurons
        list_of_out_features = [
            *literal_eval(config['hidden_size']), config.getint('output_size')]
        for out_features in list_of_out_features:
            # Add fully connected layer
            self.fc_layers.append(nn.Linear(in_features, out_features))
            # Add batchnorm layer, if desired
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_features))
            # Add dropout layer, if desired
            if self.dropout_rate:
                self.dropout_layers.append(nn.Dropout(self.dropout_rate))
            # Add activation function
            self.acts.append(nn.GELU())
            in_features = out_features
            self.num_out_features = out_features

        # Transform to pytorch list modules
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.acts = nn.ModuleList(self.acts)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fully connected neural network.

        Args:
            input_tensor: Tensor with input features

        Returns:
            Output prediction tensor
        """
        for i_layer in range(len(self.fc_layers)):
            # Fully connected layer
            input_tensor = self.fc_layers[i_layer](input_tensor)
            # Use dropout, but note after first and last layer
            if self.dropout_rate and (1 <= i_layer < len(self.fc_layers)-1):
                input_tensor = self.dropout_layers[i_layer](input_tensor)
            # Use batchnorm after each layer, but not after last
            if self.use_batch_norm and (i_layer < len(self.fc_layers)-1):
                input_tensor = self.bn_layers[i_layer](input_tensor)
            # Apply activation function, but not after last layer
            if i_layer < len(self.fc_layers)-1:
                input_tensor = self.acts[i_layer](input_tensor)
        return input_tensor


class Model:
    """
    Wrapper around neural network.

    Includes functions to train and validate network.

    Args:
        dataloader_train: Dataloader with training data
        dataloader_val: Dataloader with validation or test data
        network: PyTorch module with the network topology
        classification: If true, use cross entropy loss
    """

    def __init__(self,
                 dataloader_train: DataLoader,
                 dataloader_val: DataLoader,
                 network: nn.Module,
                 config: SectionProxy):
        super().__init__()

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.net = network

        # Use gpu if available
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print('Using:', self.device)
        self.net = self.net.to(self.device)

        if config.get_boolean('classification'):
            # Cross entropy loss function
            # Note that this includes softmax function
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            # Mean squared error loss function
            self.criterion = nn.MSELoss().to(self.device)

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.getfloat('learning_rate'))

        # Learning rate scheduler in case learning rate is too large
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config.getint('scheduler_patience'),
            factor=config.getfloat('scheduler_factor'),
            min_lr=config.getfloat('scheduler_min_lr'))

    def train(self) -> float:
        """
        Train model over one epoch.

        Returns:
            Loss averaged over the training data
        """
        self.net = self.net.train()

        sum_loss, cnt = 0, 0
        for (data, target) in self.dataloader_train:
            data = data.to(self.device)
            target = target.to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            output = self.net(data)

            # compute loss
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # measure accuracy on batch
            sum_loss += loss.detach().cpu().numpy()
            cnt += 1

        return sum_loss / cnt

    def validate(self) -> float:
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns:
            Loss averaged over the validation data
        """
        self.net = self.net.eval()

        sum_loss, cnt = 0, 0
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.dataloader_val):
            for (data, target) in self.dataloader_val:
                data = data.to(self.device)
                target = target.to(self.device)

                # forward
                output = self.net(data)

                # loss / accuracy
                sum_loss += self.criterion(output,
                                           target).detach().cpu().numpy()
                cnt += 1

        # Learning Rate reduction
        self.scheduler.step(sum_loss / cnt)

        return sum_loss / cnt

    def save_network(self, model_file_name: str) -> str:
        """
        Save model to disk.

        Args:
            model_file_namee: Path and model filename.

        Returns:
            Model filename.
        """
        torch.save(self.net.state_dict(), model_file_name)
        return model_file_name

    def load_network(self, model_file_name: str) -> None:
        """
        Load model from disk.

        Args:
            model_file_name: Path and model filename.
        """
        self.net.load_state_dict(torch.load(model_file_name))
