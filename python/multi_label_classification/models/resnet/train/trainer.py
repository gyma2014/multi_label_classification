"""resnet module
"""
import os
from .loader import ImageLabelsLoader
import torch
import torch.nn as nn
import torchvision
import time
import multi_label_classification.models.resnet.train.utils as utils


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_loaders(data_root_dir, image_size, crop_size):

    # TODO: implement the cache logic

    train_catalog = os.path.join(data_root_dir, "train.csv")
    val_catalog = os.path.join(data_root_dir,  "val.csv")

    assert os.path.isfile(train_catalog), train_catalog
    assert os.path.isfile(val_catalog), val_catalog

    train_image_dir = os.path.join(data_root_dir, "train")
    val_image_dir = os.path.join(data_root_dir, "val")

    train_loader = ImageLabelsLoader(csv_path=train_catalog,
                                     is_train=True,
                                     image_dir=train_image_dir,
                                     size=image_size,
                                     crop=crop_size,
                                     batch_size=32)

    val_loader = ImageLabelsLoader(csv_path=val_catalog,
                                   is_train=False,
                                   image_dir=val_image_dir,
                                   size=image_size,
                                   crop=crop_size,
                                   batch_size=8)

    return train_loader, val_loader


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    train_loss = 0
    num = 0

    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            output = model(image)

            # sigmoid activation function for multi-label classification
            output = torch.sigmoid(output)
            loss = criterion(output, target)
            train_loss += loss.item()
            num += 1
            loss.backward()

        optimizer.zero_grad()
        optimizer.step()
        
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return train_loss/num


def train(model, optimizer, epochs, train_loader, val_loader, device, model_dir):

    # [criterion]: multi-label loss, therefore, using BCELoss instead of Cross Entropy Loss
    criterion = nn.BCELoss()

    # [lr scheduler]: step lr used
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print("start training ...")
    
    start = time.time()

    losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "date_time": "", # TODO:
            "model_info": {}, # TODO
            "labels": train_loader.attributes, # store the labels
        }

        torch.save(checkpoint, os.path.join(model_dir, f"model_{epoch}.pth"))

        losses.append(train_loss)

        print(f"Epoch: {epoch}: {train_loss}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        plt.plot(losses, color='orange', label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./loss.png')

    total_time = time.time() - start
    import datetime
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")