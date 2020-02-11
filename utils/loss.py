import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy2d(nn.Module):
    '''
    loss doesn't change, loss can not be backward?
    '''
    def __init__(self):
        super(CrossEntropy2d, self).__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=True,
                                        ignore_index=255)# should size_average=False?

    def forward(self, out, target):
        n, c, h, w = out.size() # n:batch_size, c:class
        out = out.view(-1, c)
        target = target.view(-1)
        # print('out', out.size(), 'target', target.size())

        loss = self.criterion(out, target)

        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)