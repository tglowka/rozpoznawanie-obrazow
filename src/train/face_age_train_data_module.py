from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from commons.face_age_dataset import FaceAgeDataset


class FaceAgeTrainDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_dir,
        batch_size,
        num_workers,
        pin_memory,
        transformer_list=[],
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transformer_list = transformer_list

        self.train_indices = None
        self.val_indices = None

        self.all_train_data = None

    def set_folds(self, train_indices, val_indices):
        self.train_indices = train_indices
        self.val_indices = val_indices

    def setup(self, stage: str = None):
        transform = transforms.Compose(self.transformer_list)

        if stage == "fit" and not self.all_train_data:
            self.all_train_data = FaceAgeDataset(
                imgs_dir=self.train_data_dir, transform=transform
            )

    def train_dataloader(self):
        print(
            f"train_dataloader, min: {min(self.train_indices)}, max: {max(self.train_indices)}"
        )
        train_data = Subset(self.all_train_data, self.train_indices)
        return self.__create_data_loader(dataset=train_data, shuffle=False)

    def val_dataloader(self):
        print(
            f"val_dataloader, min: {min(self.val_indices)}, max: {max(self.val_indices)}"
        )
        val_data = Subset(self.all_train_data, self.val_indices)
        return self.__create_data_loader(dataset=val_data, shuffle=False)

    def __create_data_loader(self, dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
