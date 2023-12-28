import pytorch_lightning as pl
import torch

from test.face_age_test_data_module import FaceAgeTestDataModule
from commons.face_age_module import FaceAgeModule
from torchmetrics import MeanAbsoluteError as MAE


class FaceAgeModelTester:
    def __init__(self, net, ckpt_paths, transforms_list, debug) -> None:
        if debug:
            self.test_data_dir = "data/face_age_dataset_debug/test"
        else:
            self.test_data_dir = "data/face_age_dataset/test"

        self.ckpt_paths = ckpt_paths
        self.net = net
        self.transforms_list = transforms_list

    def __create_data_module(self):
        return FaceAgeTestDataModule(
            test_data_dir=self.test_data_dir,
            batch_size=32,
            num_workers=3,
            pin_memory=False,
            transforms_list=self.transforms_list,
        )

    def __create_model(self):
        return FaceAgeModule(net=self.net, normalize_age_by=80)

    def __create_trainer(self):
        return pl.Trainer(default_root_dir="log", callbacks=[], logger=[])

    def test(self):
        datamodule = self.__create_data_module()
        model = self.__create_model()
        trainer = self.__create_trainer()
        trainer.test(model=model, datamodule=datamodule, ckpt_path=self.ckpt_paths[0])

    def predict(self):
        preds = None
        labels = None

        datamodule = self.__create_data_module()
        for i, ckpt_path in enumerate(self.ckpt_paths):
            model = self.__create_model()
            trainer = self.__create_trainer()
            predictions = trainer.predict(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path
            )
            ckpt_preds = torch.zeros([0, 1])
            ckpt_labels = torch.zeros([0, 1])

            for batch_results in predictions:
                ckpt_preds = torch.cat((ckpt_preds, batch_results[0]))
                ckpt_labels = torch.cat((ckpt_labels, batch_results[1]))

            if i == 0:
                preds = ckpt_preds
                labels = ckpt_labels
            else:
                preds += ckpt_preds

        mean_preds = preds / len(self.ckpt_paths)
        mae = MAE()
        result = mae(mean_preds, labels)
        return result.item()
