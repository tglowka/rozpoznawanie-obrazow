import os
import random
import torch
from torch.utils.data import Dataset


class FaceAgeDataset(Dataset):
    def __init__(self, imgs_dir, transform):
        self.imgs_dir = imgs_dir
        self.normalize_age_by = 80
        self.transform = transform
        self.img_paths = []
        self.labels = []
        print("Load data!")
        self.__load_data()
        zipped = list(zip(self.img_paths, self.labels))
        random.shuffle(zipped)
        self.img_paths, self.labels = zip(*zipped)

    def __load_data(self):
        for filename in os.listdir(self.imgs_dir):
            if filename.split(".")[-1] == "pt":
                self.img_paths.append(os.path.join(self.imgs_dir, filename))
                self.labels.append(int(filename.split("_")[-1].split(".")[0]))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = torch.FloatTensor(torch.load(self.img_paths[idx]))
        label = torch.Tensor([self.labels[idx]])
        if self.normalize_age_by:
            label /= self.normalize_age_by
        if self.transform:
            img = self.transform(img)
        return img, label
