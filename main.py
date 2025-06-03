"""
This script trains a Vision Transformer (ViT) model on the TissueMNIST dataset.
It includes data loading, training, evaluation, and visualization of results.

It uses PyTorch and the MedMNIST library for dataset handling.

Written by: David Straat, Dariush Mirkarimi
Modified on: 2025-05-27
"""

import random
from typing import Union, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as t
from medmnist import TissueMNIST
from visualise import make_confusion_matrix, plot_loss, plot_accuracy, make_roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTModel, ViTConfig
import pickle

def settings():
    """
    Sets the random seed for reproducibility and configures PyTorch
    to use deterministic algorithms.
    """
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True

class Model(torch.nn.Module):
    def __init__(self, num_classes:int=5):
        super().__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        config.image_size = 64
        config.num_hidden_layers = 10
        self.vit = ViTModel(config)

        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        old_embed = model.embeddings.position_embeddings.data
        cls_embed = old_embed[:, :1, :]
        spatial_embed = old_embed[:, 1:, :]
        old_size = int(spatial_embed.shape[1] ** 0.5)
        new_size = config.image_size // config.patch_size
        spatial_embed = (spatial_embed.reshape(1, old_size, old_size, -1).
                         permute(0, 3, 1, 2))
        new_spatial_embed = torch.nn.functional.interpolate(
            spatial_embed, size=(new_size, new_size), mode='bilinear',
            align_corners=False
        ). permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        new_embed = torch.cat(
            [cls_embed, new_spatial_embed],
            dim=1
        )
        full_state_dict = model.state_dict()
        new_state_dict = self.vit.state_dict()

        for key in new_state_dict.keys():
            if key == 'embeddings.position_embeddings':
                continue
            if 'encoder.layer.' in key:
                layer_num = int(key.split('.')[2])
                if layer_num >= 10:
                    continue
            if key in full_state_dict:
                new_state_dict[key] = full_state_dict[key]

        self.vit.load_state_dict(new_state_dict)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(config.hidden_size),
            torch.nn.Linear(config.hidden_size, num_classes)
        )
        self.vit.embeddings.position_embeddings = torch.nn.Parameter(new_embed)



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: torch.Tensor of shape (batch_size, 1, 64, 64). This is a
            grayscale image tensor that will be converted to RGB by repeating
            the single channel three times.
        :type x: torch.Tensor
        :return: logits: torch.Tensor of shape (batch_size, num_classes).
        :rtype: torch.Tensor
        """
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        outputs = self.vit(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

class DatasetWrapper(Dataset):
    def __init__(self, dataset:TissueMNIST, transform=None):
        self.imgs = dataset.imgs
        self.labels = dataset.labels.squeeze()
        self.transform = transform

        mask = self.labels < 5
        self.imgs = self.imgs[mask]
        self.labels = self.labels[mask]

    def __len__(self)->int:
        return len(self.labels)

    def __getitem__(self, idx)->tuple:
        image = self.imgs[idx]
        label = self.labels[idx]

        image = torch.tensor(image).float().unsqueeze(0)/255.0
        if self.transform:
            image = self.transform(image)
        return image, int(label)

def load_data(batch_size:int=64)-> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the TissueMNIST dataset and prepares DataLoaders for training,
    validation, and testing.
    :param batch_size: the number of samples per batch to load.
    :type batch_size: int
    :raises ValueError: if batch_size is not a positive integer.
    :return: train_loader, test_loader, validation_loader: DataLoaders for
        training, testing, and validation datasets respectively.
    :rtype: DataLoader, DataLoader, DataLoader
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    train_ds = TissueMNIST(
        split='train',
        download=True,
        size=64)
    test_ds = TissueMNIST(
        split='test',
        download=True,
        size=64)
    validation_ds = TissueMNIST(
        split='val',
        download=True,
        size=64)

    transform = t.Compose([
        t.RandomHorizontalFlip(),
        # t.RandomVerticalFlip(),
        t.RandomRotation(degrees=15),
        t.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # t.Resize((64, 64)),
        # t.ToTensor(),
        # t.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = DatasetWrapper(train_ds, transform=transform)
    test_dataset = DatasetWrapper(test_ds)
    validation_dataset = DatasetWrapper(validation_ds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False)
    return train_loader, test_loader, validation_loader

def train_once(model:torch.nn.Module, data_loader:DataLoader,
              optimizer:torch.optim.Optimizer, device:torch.device,
              criterion:torch.nn.CrossEntropyLoss) -> tuple[
    float, float]:
    """
    Trains the model for one epoch on the provided data loader.
    :param model: The model to be trained.
    :param data_loader: The DataLoader providing the training data.
    :param optimizer: The optimizer to update the model parameters.
    :param device: The device (CPU or GPU) on which to perform the training.
    :param criterion: The loss function to compute the loss.
    :type model: torch.nn.Module
    :type data_loader: DataLoader
    :type optimizer: torch.optim.Optimizer
    :type device: torch.device
    :type criterion: torch.nn.CrossEntropyLoss
    :return: loss: float, accuracy: float. The average loss and accuracy
        over the epoch.
    :rtype: float and float
    """
    model.train()
    loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        optimizer.zero_grad()
        outputs = model(images)
        loss_value = criterion(outputs, labels)
        loss_value.backward()
        optimizer.step()
        loss += loss_value.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
    import time
    return loss / total, correct / total

def eval_once(model:torch.nn.Module, data_loader:DataLoader,
              device:torch.device, criterion:torch.nn.CrossEntropyLoss) -> (
        tuple[float, float]):
    """
    Evaluates the model on the provided data loader.
    :param model: The model to be evaluated.
    :param data_loader: The DataLoader providing the evaluation data.
    :param device: The device (CPU or GPU) on which to perform the evaluation.
    :param criterion: The loss function to compute the loss.
    :type model: torch.nn.Module
    :type data_loader: DataLoader
    :type device: torch.device
    :type criterion: torch.nn.CrossEntropyLoss
    :return: The average loss and accuracy over the evaluation dataset.
    :rtype: float, float
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            loss += loss_value.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()

    return loss / total, correct / total

def main():
    settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_load, test_load, validation_load = load_data(batch_size=32)
    print(f'Train dataset size: {len(train_load.dataset)}')
    model = Model(num_classes=5).to(device)
    print(f'Model initialized with '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} '
          f'trainable parameters.')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 5
    print(f'Starting training for {num_epochs} epochs...')
    train_loss, train_accuracy, val_loss, val_accuracy = [], [], [], []
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        train_epoch_loss, train_epoch_accuracy = train_once(
            model, train_load, optimizer, device, criterion)
        val_epoch_loss, val_epoch_accuracy = eval_once(
            model, validation_load, device, criterion)

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        import time
        with open(f'model{int(time.time())}.pkl', 'wb') as f:
            pickle.dump(model.state_dict(), f)
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_epoch_loss:.4f}, '
              f'Train Accuracy: {train_epoch_accuracy:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, '
              f'Val Accuracy: {val_epoch_accuracy:.4f}')
    with open("values.pkl", 'wb') as f:
        pickle.dump({
            "train_loss": train_loss,
            "val_loss" : val_loss,
            "num_epochs" : num_epochs,
            "train_accuracy" : train_accuracy,
            "val_accuracy" : val_accuracy,
            "num_epochs" : num_epochs,
            "model": model,
            "test_load": test_load,
            "device": device,
            "criterion": criterion,
            "classes": classes
        }, f)
    plot_loss(train_loss, val_loss, num_epochs)
    plot_accuracy(train_accuracy, val_accuracy, num_epochs)

    test_loss, test_accuracy = eval_once(
        model, test_load, device, criterion
    )
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    make_confusion_matrix(model, test_load, device, classes)
    make_roc_curve(model, test_load, device, num_classes=5)

if __name__ == '__main__':
    main()