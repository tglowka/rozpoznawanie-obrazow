from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from commons.face_age_dataset import FaceAgeDataset


class FaceAgeTestDataModule(LightningDataModule):
    def __init__(
        self,
        test_data_dir,
        batch_size,
        num_workers,
        pin_memory,
        transforms_list=[],
    ):
        super().__init__()
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms_list = transforms_list

        self.test_data = None

    def setup(self, stage: str = None):
        transform = transforms.Compose(self.transforms_list)

        if stage == "predict" and not self.test_data:
            self.test_data = FaceAgeDataset(
                imgs_dir=self.test_data_dir,
                transform=transform,
            )

        if stage == "test" and not self.test_data:
            self.test_data = FaceAgeDataset(
                imgs_dir=self.test_data_dir,
                transform=transform,
            )

    def predict_dataloader(self):
        return self.__create_data_loader(dataset=self.test_data, shuffle=False)

    def test_dataloader(self):
        return self.__create_data_loader(dataset=self.test_data, shuffle=False)

    def __create_data_loader(self, dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )
