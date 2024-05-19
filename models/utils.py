import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    accuracy, precision, recall, auroc, average_precision, specificity
)


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y, reduction='none')
        pt = torch.exp(-loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        return F_loss.mean()


def get_metrics(y_hat, y, num_classes):
    """using ver torchmetrics==0.11.1"""
    if y_hat.shape[1] > 2:
        y_argmax = torch.argmax(y, axis=1)
        y_hat_argmax = torch.argmax(y_hat, axis=1)
        acc_macro_avg = accuracy(y_hat_argmax, y_argmax, average='macro', num_classes=num_classes, task='multiclass')
        acc_micro_avg = accuracy(y_hat_argmax, y_argmax, average='micro', num_classes=num_classes, task='multiclass')
        acc_weighted_avg = accuracy(y_hat_argmax, y_argmax, average='weighted', num_classes=num_classes, task='multiclass')
        precision_macro_avg = precision(y_hat_argmax, y_argmax, average='macro', num_classes=num_classes, task='multiclass')
        precision_micro_avg = precision(y_hat_argmax, y_argmax, average='micro', num_classes=num_classes, task='multiclass')
        precision_weighted_avg = precision(y_hat_argmax, y_argmax, average='weighted', num_classes=num_classes, task='multiclass')
        recall_macro_avg = recall(y_hat_argmax, y_argmax, average='macro', num_classes=num_classes, task='multiclass')
        recall_micro_avg = recall(y_hat_argmax, y_argmax, average='micro', num_classes=num_classes, task='multiclass')
        recall_weighted_avg = recall(y_hat_argmax, y_argmax, average='weighted', num_classes=num_classes, task='multiclass')
        specificity_macro_avg = specificity(y_hat_argmax, y_argmax, average='macro', num_classes=num_classes, task='multiclass')
        specificity_micro_avg = specificity(y_hat_argmax, y_argmax, average='micro', num_classes=num_classes, task='multiclass')
        specificity_weighted_avg = specificity(y_hat_argmax, y_argmax, average='weighted', num_classes=num_classes, task='multiclass')
        auroc_macro_avg = auroc(y_hat, y_argmax, task='multiclass', num_classes=num_classes, average='macro')
        auroc_weighted_avg = auroc(y_hat, y_argmax, task='multiclass', num_classes=num_classes, average='weighted')
        pr_auc_macro_avg = average_precision(y_hat, y_argmax, task='multiclass', num_classes=num_classes, average='macro')
        pr_auc_weighted_avg = average_precision(y_hat, y_argmax, task='multiclass', num_classes=num_classes, average='weighted')

        return {
            'acc_macro_avg': acc_macro_avg,
            'acc_micro_avg': acc_micro_avg,
            'acc_weighted_avg': acc_weighted_avg,
            'precision_macro_avg': precision_macro_avg,
            'precision_micro_avg': precision_micro_avg,
            'precision_weighted_avg': precision_weighted_avg,
            'recall_macro_avg': recall_macro_avg,
            'recall_micro_avg': recall_micro_avg,
            'recall_weighted_avg': recall_weighted_avg,
            'specificity_macro_avg': specificity_macro_avg,
            'specificity_micro_avg': specificity_micro_avg,
            'specificity_weighted_avg': specificity_weighted_avg,
            'auroc_macro_avg': auroc_macro_avg,
            'auroc_weighted_avg': auroc_weighted_avg,
            'pr_auc_macro_avg': pr_auc_macro_avg,
            'pr_auc_weighted_avg': pr_auc_weighted_avg,
        }
    elif y_hat.shape[1] == 2:
        pos_index = 1
        y = torch.argmax(y, axis=1)
        y_hat = F.softmax(y_hat, dim=1)[:, pos_index]
        acc = accuracy(y_hat, y, task='binary')
        roc_auc = auroc(y_hat, y, task='binary')
        precision_ = precision(y_hat, y, task='binary')
        recall_ = recall(y_hat, y, task='binary')
        pr_auc = average_precision(y_hat, y, task='binary')
        return {
            'acc': acc,
            'roc_auc': roc_auc,
            'precision': precision_,
            'recall': recall_,
            'pr_auc': pr_auc
        }
    else:
        raise Exception('error')
