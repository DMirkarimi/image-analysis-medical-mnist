import torch
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import numpy as np

def make_confusion_matrix(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader,
                             device:torch.device, classes:list):
    """
    Generates and displays a confusion matrix for the model's predictions
    on the provided data loader.
    This function evaluates the model on the data loader, collects the
    predictions and true labels, and then computes the confusion matrix.
    :param model: The model to be evaluated.
    :param data_loader: The DataLoader providing the evaluation data.
    :param device: The device (CPU or GPU) on which to perform the evaluation.
    :param classes: The list of class names for the confusion matrix.
    :type model: torch.nn.Module
    :type data_loader: torch.utils.data.DataLoader
    :type device: torch.device
    :type classes: list

    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(predicted)
            all_preds.extend(labels.nump())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def make_roc_curve(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader,
              device:torch.device, num_classes:int):
    """
    Generates and displays the ROC curve for each class in a multi-class
    classification problem. This function evaluates the model on the provided
    data loader, collects the predicted scores and true labels, and then
    computes the ROC curve and AUC for each class.
    :param model: The model to be evaluated.
    :param data_loader: The DataLoader providing the evaluation data.
    :param device: The device (CPU or GPU) on which to perform the evaluation.
    :param num_classes: The number of classes in the classification problem.
    :type model: torch.nn.Module
    :type data_loader: torch.utils.data.DataLoader
    :type device: torch.device
    :type num_classes: int
    """
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    labels = label_binarize(all_labels, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], all_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_loss(train_loss:list, val_loss:list, num_epochs:int):
    """
    Plots and displays the training and validation loss curves.
    :param train_loss: List of training loss values for each epoch.
    :param val_loss: List of validation loss values for each epoch.
    :param num_epochs: The total number of epochs.
    :type train_loss: list
    :type val_loss: list
    :type num_epochs: int
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy:list, val_accuracy:list, num_epochs:int):
    """
    Plots and displays the training and validation accuracy curves.
    :param train_accuracy: 
    :param val_accuracy:
    :param num_epochs:
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, num_epochs + 1),
        train_accuracy,
        label='Train Accuracy')
    plt.plot(
        range(1, num_epochs + 1),
        val_accuracy,
        label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()

    