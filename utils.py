import torch
import torch.nn as nn


def progress(train_loss: float, val_loss: float) -> str:
    """
    Create progress bar description.

    Arguments
    -------
    train_loss     - Training loss
    val_loss       - Validation or test loss

    Returns
    -------
    String with training and test loss
    """
    return "Train/Loss: {:.8f} " \
           "Val/Loss: {:.8f}" \
           .format(train_loss, val_loss)


class DenseStack(torch.nn.Module):
    """
    Fully connected neural network.

    Arguments
    -------
    num_in_features     - Number of input features
    num_out_features    - Number of output features
    num_hidden_features - List of nodes in each hidden layer
    use_batch_norm      - If to use batch norm
    dropout_rate        - If, and with which rate, to use dropout
    """

    def __init__(self, num_in_features: int, num_out_features: int,
                 num_hidden_features: list,
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
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
            self.acts.append(nn.GELU())
            in_features = out_features
            self.num_out_features = out_features

        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.acts = nn.ModuleList(self.acts)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fully connected neural network.

        Arguments
        -------
        input_tensor        - Tensor with input features

        Returns
        -------
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

    def train(self) -> float:
        """
        Train model over one epoch.

        Returns
        -------
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

    def validate(self) -> float:
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns
        -------
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

    def save_network(self, name: str) -> str:
        """
        Save model to disk.

        Arguments
        -------
        name         - Model filename.

        Returns
        -------
        Model filename.
        """
        model_file_name = self.base_path+name
        torch.save(self.net.state_dict(), model_file_name)
        return name

    def load_network(self, name: str) -> None:
        """
        Load model from disk.

        Arguments
        -------
        name         - Model filename.
        """
        model_file_name = self.base_path+name
        self.net.load_state_dict(torch.load(model_file_name))
