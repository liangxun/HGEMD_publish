import torch
import torch.nn as nn
import numpy as np


class FGSM:
    def __init__(self, model, max_bit=10, is_cuda=False):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda

        self.loss_func = nn.CrossEntropyLoss()
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()
    
    def generate(self, data, data_grad):
        data = data.data.numpy().squeeze()
        data_grad = data_grad.numpy().squeeze()
        mask_data = (data == 0)
        mask_grad = (data_grad > 0)
        mask = mask_data & mask_grad
        data_grad[~mask] = 0

        index = data_grad.argsort()[-self.max_bit:]
        data[index] = 1
        data = torch.Tensor(data).unsqueeze(0)
        return data
    
    def attack(self, id, x, true_y):
        """
        return: 
            0   Malware is misclassified as goodware even no attack. we don’t bother to craft adversarial sample for it.
            -1  failed attack
            n(n>0)  successful attack, n is perturbation distance.
        """
        # x = x.cuda()
        x.requires_grad = True
        out = self.model.predict(id, x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item():
            # print("no need! init_pred={}, true_y={}".format(init_pred.item(), true_y.item()))
            return 0, None
        loss = self.loss_func(out, true_y.cuda() if self.is_cuda else true_y)
        self.model.zero_grad
        loss.backward()
        x_grad = x.grad.data
        adv_x = self.generate(x.cpu(), x_grad.cpu())
        out = self.model.predict(id, adv_x)
        adv_pred = out.cpu().max(1, keepdim=True)[1]
        if adv_pred.item() != true_y.item():
            # print("success!")
            return self.max_bit, adv_x.cpu().data.numpy().squeeze()
        else:
            # print("fialed!")
            return -1, None
