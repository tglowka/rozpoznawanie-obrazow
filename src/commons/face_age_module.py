from enum import Enum
import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric
from efficientnet_pytorch import EfficientNet


class Nets(Enum):
    SimpleConvNet_224x224 = 1
    PretrainedEfficientNet = 2


class PretrainedEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return x


class SimpleConvNet_224x224(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FaceAgeModule(LightningModule):
    def __init__(self, net, normalize_age_by, learning_rate=None):
        super().__init__()

        self.normalize_age_by = normalize_age_by

        if net == Nets.SimpleConvNet_224x224:
            self.net = SimpleConvNet_224x224()
        elif net == Nets.PretrainedEfficientNet:
            self.net = PretrainedEfficientNet()
        else:
            raise ValueError("Unknown net.")

        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

        self.train_mae = MAE()
        self.train_loss = MeanMetric()

        self.test_mae = MAE()
        self.test_loss = MeanMetric()

        self.val_mae = MAE()
        self.val_loss = MeanMetric()

        self.val_mae_best = MinMetric()
        self.val_mae_list = []

    def forward(self, input: torch.Tensor):
        return self.net(input)

    def predict_step(self, batch):
        _, predictions, targets = self.__model_step(batch)
        return predictions, targets

    def on_train_start(self):
        self.val_mae_best.reset()

    def __model_step(self, batch):
        input, targets = batch

        predictions = self.forward(input)
        loss = self.criterion(predictions, targets)
        # clip predictions to [0-1]
        predictions = predictions.clip(0, 1)
        # rescale predictions and labels from [0-1] to [0-80]
        if self.normalize_age_by:
            predictions = predictions * self.normalize_age_by
            predictions = predictions.clip(1, self.normalize_age_by)
            targets = targets * self.normalize_age_by

        return loss, predictions, targets

    def training_step(self, batch, batch_idx):
        loss, predictions, targets = self.__model_step(batch)
        self.train_loss(loss)
        self.train_mae(predictions, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_step(self, batch):
        loss, predictions, targets = self.__model_step(batch)
        self.val_loss(loss)
        self.val_mae(predictions, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_mae = self.val_mae.compute()
        self.val_mae_best(val_mae)
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

        self.val_mae_list.append(val_mae.item())
        print(
            "\nEpoch: "
            + str(self.current_epoch)
            + ", val_mae: "
            + str(val_mae.item())
            + "\n"
        )

    def test_step(self, batch):
        loss, predictions, targets = self.__model_step(batch)
        self.test_loss(loss)
        self.test_mae(predictions, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_end(self) -> None:
        print(self.val_mae_list)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
