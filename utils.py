import torch
import torch.nn as nn


class ListModule(nn.Module):
    """
    For creating PyTorch modules from lists.
    """

    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        iterator = iter(self._modules.values())
        for _ in range(idx):
            next(iterator)
        return next(iterator)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def progress(train_loss, val_loss):
    return "Train/Loss: {:.8f} " \
           "Val/Loss: {:.8f}" \
           .format(train_loss, val_loss)


class Swish(nn.Module):
    """
    Nonlinear swishactivation function.
    """

    def forward(self, input_tensor):
        """Forward pass through activation function."""
        return input_tensor * torch.sigmoid(input_tensor)


class DenseStack(torch.nn.Module):
    """
    Fully connected neural network composed of three layers.
    The first two layers are followed by a Swish activation function.

    Arguments:
    num_in_features (int)     - Number of input features
    num_out_features (int)    - Number of output features
    num_hidden_features (int) - Number of nodes in each hidden layer (optinal, deault: 64)
    """

    def __init__(self, num_in_features, num_out_features, num_hidden_features,
                 use_batch_norm=False, dropout_rate=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        self.fc_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        self.acts = []

        in_features = num_in_features
        for out_features in [*num_hidden_features, num_out_features]:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_features))
            if dropout_rate:
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            self.acts.append(Swish())
            in_features = out_features
            self.num_out_features = out_features

        self.fc_layers = ListModule(*self.fc_layers)
        self.bn_layers = ListModule(*self.bn_layers)
        self.dropout_layers = ListModule(*self.dropout_layers)
        self.acts = ListModule(*self.acts)

    def forward(self, input_tensor):
        """Forward pass through dense stack."""
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

    Arguments:
    dataloader_train          - Dataloader with training data
    dataloader_val            - Dataloader with validation or test data
    network                   - PyTorch module with the network topology
    classification            - If true, use cross entropy loss
    path                      - Path where the model should be saved
    """

    def __init__(self, dataloader_train, dataloader_val, network, classification=True, path=None):
        super().__init__()
        self.base_path = path

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

        self.learning_rate = 0.01

        if classification:
            # Cross entropy loss function
            # Note that this includes softmax function
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            # Mean squared error loss function
            self.criterion = nn.MSELoss().to(self.device)

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate)

        # Learning rate scheduler in case learning rate is too large
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=25, factor=0.5, min_lr=1e-7)

    def train(self):
        """
        Train model over one epoch.

        Returns
        -------
        avg_loss: float
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
            sum_loss += loss
            cnt += 1

        return sum_loss / cnt

    def validate(self):
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns
        -------
        avg_loss: float
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
                sum_loss += self.criterion(output, target)
                cnt += 1

        # Learning Rate reduction
        self.scheduler.step(sum_loss / cnt)

        return sum_loss / cnt

    def save_network(self, name):
        """
        Save model to disk.

        Arguments
        -------
        name: str
            Model filename.

        Returns
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        torch.save(self.net.state_dict(), model_file_name)
        return name

    def load_network(self, name):
        """
        Load model from disk.

        Arguments
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        self.net.load_state_dict(torch.load(model_file_name))
