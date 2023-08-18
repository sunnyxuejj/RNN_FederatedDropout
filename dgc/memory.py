import torch

__all__ = ['Memory', 'DGCSGDMemory']


# code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/memory/dgc.py
class Memory:
    @staticmethod
    def initialize(*args, **kwargs):
        pass

    @staticmethod
    def compensate(tensor, *args, **kwargs):
        return tensor

    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def state_dict():
        return None

    @staticmethod
    def load_state_dict(state_dict):
        pass


class DGCSGDMemory(Memory):
    """ Memory for momentum correction in DGC for momentum SGD optimizer"""

    def __init__(self, momentum=0, nesterov=False,
                 gradient_clipping=None, momentum_masking=True):
        self.gradient_clipping = gradient_clipping
        self.momentum_masking = momentum_masking

        self.momentum = momentum
        self.nesterov = nesterov
        self.momentums = {}
        self.velocities = {}


    def initialize(self, named_parameters):
        for name in named_parameters:
            if name != 'rnns.0.module.weight_hh_l0' and name != 'rnns.1.module.weight_hh_l0' and name != 'rnns.0.module.weight_hh_l0_raw' and name != 'rnns.1.module.weight_hh_l0_raw':
                param = named_parameters[name]
                self.momentums[name] = torch.zeros_like(param.data)
                self.velocities[name] = torch.zeros_like(param.data)

    def compensate(self, grad, name, accumulate=True):
        """Update the velocities with the momentums."""
        if self.gradient_clipping is not None:
            grad = self.gradient_clipping(grad)
        mmt = self.momentums[name]
        if accumulate:
            vec = self.velocities[name]
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                vec.add_(mmt).add_(grad)
            else:
                #mmt.mul_(self.momentum).add_(grad)
                #vec.add_(mmt)
                vec.add_(grad)
            return vec
        else:
            if self.nesterov:
                mmt.add_(grad).mul_(self.momentum)
                return mmt.add(grad)
            else:
                mmt.mul_(self.momentum).add_(grad)
                return mmt.clone()  # TODO: save this clone

    def update(self, name, ctx):
        """Update the momentums."""
        if name != 'rnns.0.module.weight_hh_l0' and name != 'rnns.1.module.weight_hh_l0' and name != 'rnns.0.module.weight_hh_l0_raw' and name != 'rnns.1.module.weight_hh_l0_raw':
            indices = ctx[0]
            if self.momentum_masking:
                self.momentums[name].view(-1).index_fill_(0, indices, 0)
            self.velocities[name].view(-1).index_fill_(0, indices, 0)

    def state_dict(self):
        return dict(momentums=self.momentums, velocities=self.velocities)

    def load_state_dict(self, state_dict):
        momentums = state_dict['momentums']
        velocities = state_dict['velocities']
        for name in self.momentums.keys():
            if name in momentums:
                self.momentums[name] = momentums[name]
                self.velocities[name] = velocities[name]
