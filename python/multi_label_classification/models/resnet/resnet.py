"""Resnet implementation
"""
import os
import torch
from torchvision import models as models

from multi_label_classification.models.resnet.train.trainer import get_loaders, train
from multi_label_classification.models.model import MultiLabelModel
import multi_label_classification.models.resnet.train.utils as utils
import torch.nn as nn


class ResNet(MultiLabelModel):
    def __init__(self, labels, model_path=None, image_width=400, image_height=400) -> None:
        super().__init__()

        self._size = (image_height, image_width)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._checkpoint = None

        if model_path and os.path.isfile(model_path):
            self._checkpoint = torch.load(model_path)
            self._model.load_state_dict(self, self._checkpoint['model_state_dict'])
            # assert the labels and checkpoint['labels']
        else:
            # passing labels as num_classes to finetune fc layers
            self._model = models.__dict__["resnet50"](pretrained=True)
            self._model.fc = nn.Linear(2048, labels)

        self._model = self._model.to(self._device)


    def fit_impl(self, data_dir, model_dir, crop=(380, 380), epochs=100, lr=0.0001, weight_decay=1e-4, ):
        """train a resnet model

        Args:
            data (_type_): A data loader
        """

        # TODO: eval against SGD and AdamW
        params = utils.set_weight_decay(self._model, weight_decay)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        train_loader, val_loader = get_loaders(data_dir, self._size, crop)
        train(self._model, optimizer, epochs, train_loader, val_loader, self._device, model_dir)
        

    def predict_impl(self, X):
        
        self._model.eval()

        # TODO: return a list of features  and its probas


if __name__ == '__main__':
    resnet = ResNet(25)

    resnet.fit(data_dir=r"C:\Users\gyma2\Documents\data\movies\Multi_Label_dataset",
               model_dir="./")

