import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_seg_contour import Parser
from utils.tools import get_model_list
from utils.logger import get_logger
import numpy as np


logger = get_logger()

class Seg_Trainer(nn.Module):
    def __init__(self, config):
        super(Seg_Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.class_num = 17      # (16 components + backgorund for CelebA-HQ)

        # Parser network
        self.netParser = Parser(self.config, self.use_cuda, self.device_ids)
        self.optimizerSGD = torch.optim.SGD(self.netParser.parameters(), lr=self.config['lr'],
                                         momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        if self.use_cuda:
            self.netParser.to(self.device_ids[0])



    def forward(self, x, ground_truth):
        self.train()
        losses = {}

        # cross entropy loss
        self.optimizerSGD.zero_grad()
        predict = self.netParser(x)

        loss = self.CrossEntropyLoss2d(predict, ground_truth)
        return loss, predict


    def save_model(self, checkpoint_dir, iteration):
        parser_name = os.path.join(checkpoint_dir, 'parser_%08d.pt' % iteration)
        parser_opt_name = os.path.join(checkpoint_dir, 'parser_optimizer.pt')

        torch.save(self.netParser.state_dict(), parser_name)
        torch.save(self.optimizerSGD, parser_opt_name)


    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "parser", iteration=iteration)
        self.netParser.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        '''
        if not test:
            # Load optimizers

            #last_opt_name = torch.load(os.path.join(checkpoint_dir, 'parser_optimizer.pt'))

            opt_state_dict = torch.load(os.path.join(checkpoint_dir, 'parser_optimizer.pt'))
            self.optimizerSGD.load_state_dict(opt_state_dict)
        '''

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration


    # for check the segmentation result with color
    def get_parsing_labels(self):
        # background + 16 components
        return np.asarray([[0, 0, 0],
                          [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                          [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                          [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128,128],
                          [0, 64, 0]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        # pick pallete table from above and paint each values
        for i, label in enumerate(self.get_parsing_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)

        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colors = self.get_parsing_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        for l in range(0, self.class_num):
            r[temp == l] = label_colors[l, 0]
            g[temp == l] = label_colors[l, 1]
            b[temp == l] = label_colors[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb



    #inputs
    def CrossEntropyLoss2d(self, inputs, targets, weight=None):
        n, c, h, w = inputs.size()
        log_p = F.log_softmax(inputs, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)  # (n*h*w, c)
        log_p = log_p[targets.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = targets >= 0
        targets = targets[mask]
        loss = F.nll_loss(log_p, targets, weight=weight)
        return loss

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist


    def scores(self, label_trues, label_preds, n_class):
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(n_class), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu
