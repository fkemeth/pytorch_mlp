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
                 use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        fc_layers = []
        bn_layers = []
        acts = []

        in_features = num_in_features
        for out_features in [*num_hidden_features, num_out_features]:
            fc_layers.append(nn.Linear(in_features, out_features))
            bn_layers.append(nn.BatchNorm1d(out_features))
            acts.append(Swish())
            in_features = out_features
            self.num_out_features = out_features

        self.fc_layers = ListModule(*fc_layers)
        self.bn_layers = ListModule(*bn_layers)
        self.acts = ListModule(*acts)

    def forward(self, input_tensor):
        """Forward pass through dense stack."""
        for fully_connect, batch_norm, activation in zip(self.fc_layers, self.bn_layers, self.acts):
            input_tensor = fully_connect(input_tensor)
            if self.use_batch_norm:
                input_tensor = batch_norm(input_tensor)
            input_tensor = activation(input_tensor)
        return input_tensor


class Model:
    def __init__(self, dataloader_train, dataloader_val, network, config, path):
        super().__init__()
        self.base_path = path

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.net = network
        self.device = self.net.device
        print('Using:', self.device)
        self.net = self.net.to(self.device)

        self.learning_rate = float(config["lr"])

        self.criterion = nn.MSELoss(reduction='sum').to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=int(config["patience"]),
            factor=float(config["reduce_factor"]), min_lr=1e-7)

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
        for (data, delta_x, target, param) in self.dataloader_train:
            data = data.to(self.device)
            delta_x = delta_x.to(self.device)
            target = target.to(self.device)
            if self.net.use_param:
                param = param.to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            if self.net.use_param:
                output = self.net(data, delta_x, param)
            else:
                output = self.net(data, delta_x)

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
            for (data, delta_x, target, param) in self.dataloader_val:
                data = data.to(self.device)
                delta_x = delta_x.to(self.device)
                target = target.to(self.device)
                if self.net.use_param:
                    param = param.to(self.device)

                # forward
                if self.net.use_param:
                    output = self.net(data, delta_x, param)
                else:
                    output = self.net(data, delta_x)

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
