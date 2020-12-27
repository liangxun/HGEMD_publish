import torch
import numpy as np


class JSMA:
    def __init__(self, model, max_bit=10, is_cuda=False):
        self.model = model
        self.max_bit = max_bit
        self.is_cuda = is_cuda
        self.model.eval()
        if self.is_cuda:
            self.model = self.model.cuda()
    
    def generate(self, data, data_grad):
        data = data.data.numpy().squeeze()
        data_grad_0 = data_grad[0].numpy().squeeze()
        data_grad_1 = data_grad[1].numpy().squeeze()
        assert data.shape[0] == data_grad_0.shape[0] == data_grad_1.shape[0]

        s_map_0 = np.zeros_like(data)
        for i in range(s_map_0.shape[0]):
            if data_grad_0[i] < 0 or data_grad_1[i] > 0:
                pass
            else:
                s_map_0[i] = data_grad_0[i] * abs(data_grad_1[i])

        mask_data = (data == 0)
        mask_smap = (s_map_0 > 0)
        mask = mask_data & mask_smap
        s_map_0[~mask] = 0
        # modify on bit
        index = s_map_0.argsort()[-1]
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
        x.requires_grad = True
        out = self.model.predict(id, x)
        init_pred = out.cpu().max(1, keepdim=True)[1]
        if init_pred.item() != true_y.item(): 
            # print("no need! init_pred={}, true_y={}".format(init_pred.item(), true_y.item()))
            return 0, None

        for i in range(self.max_bit):
            self.model.zero_grad()
            out[0][0].backward(retain_graph=True)
            data_grad_0 = x.grad.data.clone()
            out[0][1].backward()
            data_grad_1 = x.grad.data.clone()
            data_grad = (data_grad_0.cpu(), data_grad_1.cpu())
            adv_x = self.generate(x.cpu(), data_grad)
            x = adv_x
            x.requires_grad = True
            out = self.model.predict(id, x)
            adv_pred = out.cpu().max(1, keepdim=True)[1]
            if adv_pred.item() != init_pred.item():
                # print("success! {} iters".format(i+1))
                return i+1, adv_x.cpu().data.numpy().squeeze()
        # print("failed! max up {} iters".format(i+1))
        return -1, None
