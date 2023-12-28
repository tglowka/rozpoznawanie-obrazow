import os
import pytorch_lightning as pl
from sklearn.model_selection import KFold

from train.face_age_train_data_module import FaceAgeTrainDataModule
from commons.face_age_module import FaceAgeModule


class FaceAgeModelFitter:
    def __init__(
        self, folds, net, epoch, transformer_list, debug, learning_rate
    ) -> None:
        if debug:
            self.train_data_dir = "data/face_age_dataset_debug/train"
        else:
            self.train_data_dir = "data/face_age_dataset/train"

        self.folds = folds
        self.net = net
        self.epoch = epoch
        self.transformer_list = transformer_list
        self.learning_rate = learning_rate

    def __create_folds(self):
        total_count = len(os.listdir(self.train_data_dir))
        kfold = KFold(n_splits=self.folds)
        return kfold.split(list(range(total_count)))

    def __create_data_module(self):
        return FaceAgeTrainDataModule(
            train_data_dir=self.train_data_dir,
            batch_size=32,
            num_workers=3,
            pin_memory=False,
            transformer_list=[],
        )

    def __create_model(self):
        return FaceAgeModule(
            net=self.net, normalize_age_by=80, learning_rate=self.learning_rate
        )

    def __create_trainer(self):
        callbacks = []
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                dirpath="log/checkpoints",
                save_top_k=1,
                save_last=True,
                mode="min",
                save_weights_only=True,
                filename="best-checkpoint",
            )
        )
        return pl.Trainer(
            default_root_dir="log",
            callbacks=callbacks,
            logger=[],
            max_epochs=self.epoch,
            num_sanity_val_steps=0,
        )

    def fit(self):
        fold_splits = self.__create_folds()
        datamodule = self.__create_data_module()

        for i, (train_indices, val_indices) in enumerate(fold_splits):
            datamodule.set_folds(train_indices, val_indices)
            model = self.__create_model()
            trainer = self.__create_trainer()
            trainer.fit(model=model, datamodule=datamodule)
