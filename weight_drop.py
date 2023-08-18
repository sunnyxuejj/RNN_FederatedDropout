import torch
from torch.nn import Parameter
import numpy as np

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            #del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))  # autograd实现任意标量值函数的自动微分
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training))
            #print(w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self.masks = self._setweights()
        return self.module.forward(*args)

class WeightCompress(torch.nn.Module):
    def __init__(self, module, weights, mask):
        super(WeightCompress, self).__init__()
        self.module = module
        self.weights = weights
        self.mask = mask

    def _setweights(self):
        for name_w in self.weights:
            #print('reshape weight drop of {}'.format(name_w))
            w = getattr(self.module, name_w)
            w = w.detach().numpy()
            new_w = []
            for i in range(w.shape[0]):
                if self.mask[i] == False:
                    continue
                new_w.append(w[i])
            new_w = torch.nn.Parameter(torch.tensor(new_w))
            setattr(self.module, name_w, new_w)

    def forward(self):
        self._setweights()
        return self.module

class WeightRecover(torch.nn.Module):
    def __init__(self, module, weights, mask):
        super(WeightRecover, self).__init__()
        self.module = module
        self.weights = weights
        self.mask = mask

    def _setweights(self):
        for name_w in self.weights:
            print('recover weight of {}'.format(name_w))
            w = getattr(self.module, name_w)
            w = w.detach().numpy()
            new_w = []
            j = 0
            for i in range(self.mask.shape[0]):
                if self.mask[i]:
                    new_w.append(w[j])
                    j = j + 1
                else:
                    new_w.append(np.zeros(w.shape[1]))
            new_w = torch.nn.Parameter(torch.tensor(new_w))
            setattr(self.module, name_w, new_w)

    def forward(self):
        self._setweights()
        return self.module

if __name__ == '__main__':
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).to(device)
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    #lin = lin.to(device)
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.5, variational=True)
    #wdrnn = wdrnn.to(device)

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')
