"""data loader
"""
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from .presets import ClassificationPresetTrain, ClassificationPresetEval
from PIL import Image


class ImageLabelsLoader(DataLoader):

    class _ImageDataset(Dataset):
        def __init__(self, csv, train, image_dir, size, crop) -> None:
            super().__init__()
            self._csv = csv
            self._is_train = train
            self._dir = image_dir

            assert os.path.isdir(self._dir), f'{self._dir} is not directory'

            self._size = size
            self._crop = crop

            if train:
                self._transforms = ClassificationPresetTrain(resize_size=self._size, crop_size=self._crop)
            else:
                self._transforms = ClassificationPresetEval(resize_size=self._size, crop_size=self._crop)

            self._target_transforms = transforms.Compose([transforms.ToTensor()])

            reader = pd.read_csv(self._csv)
            self._attributes = reader.columns[1:]
            self._data = reader.to_numpy()

        def __len__(self):
            return len(self._data)

        @property
        def attributes(self):
            return self._attributes

        def __getitem__(self, index):
            
            image_path = os.path.join(self._dir, self._data[index][0])
            assert os.path.isfile(image_path), image_path

            with open(image_path, 'rb') as fp:
                image = Image.open(fp).convert("RGB")
            labels = list(self._data[index][1:])


            return self._transforms(image),   torch.tensor(labels, dtype=torch.float32)


    def __init__(self, csv_path, is_train, image_dir, size, crop, batch_size):

        dataset = self._ImageDataset(csv_path, is_train, image_dir, size, crop)

        super(ImageLabelsLoader, self).__init__(dataset=dataset,
                                                batch_size=batch_size, 
                                                shuffle=True if is_train else False,
                                                pin_memory=True)

    @property
    def attributes(self):
        return self.dataset.attributes
