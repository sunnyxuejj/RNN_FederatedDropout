import math
import random

import torch
from dgc.memory import DGCSGDMemory

__all__ = ['DGCCompressor']


class DGCCompressor:
    def __init__(self, compress_ratio=0.75, memory=None,
                 sample_ratio=0.01, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.8, max_adaptation_iters=10, resample=True,
                 fp16_values=False, int32_indices=False,
                 warmup_epochs=21, warmup_coeff=None):
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = DGCSGDMemory() if memory is None else memory
        self.warmup_epochs = warmup_epochs
        self.warmup_coeff = [0.75, 0.9375, 0.984375, 0.996, 0.999]
        # self.warmup_coeff = [0.25, 0.0625, 0.015625, 0.004, 0.001]

        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}

    def initialize(self, named_parameters):
        for name in named_parameters.keys():
            param = named_parameters[name]
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            if self.sample_ratio < 1.0:
                pct_numel = int(math.ceil(numel * self.sample_ratio))
                cpr_numel = int(math.ceil(2 / self.compress_ratio))
                if numel <= cpr_numel:
                    sample_stride = 1
                    num_samples = numel
                else:
                    sample_stride = int(math.ceil(numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1
                    num_samples = numel // sample_stride
                    while num_samples < max(pct_numel, cpr_numel):
                        sample_stride = sample_stride - 8
                        num_samples = numel // sample_stride
            else:
                sample_stride = 1
                num_samples = numel
            if name != 'rnns.0.module.weight_hh' and name != 'rnns.1.module.weight_hh':
                top_k_samples = int(math.ceil(num_samples * (1 - self.compress_ratio)))
                num_selects = int(math.ceil(numel * (1 - self.compress_ratio)))
            else:
                top_k_samples = int(math.ceil(num_samples * 0.75))
                num_selects = int(math.ceil(numel * 0.75))
            self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)

    def warmup_compress_ratio(self, epoch, named_parameters):
        if self.warmup_epochs > 0:
            if epoch and epoch % 5 == 0 and epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[int(epoch / 5)]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                         self.base_compress_ratio)
            else:
                compress_ratio = self.compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            self.compress_ratio = compress_ratio
            self.initialize(named_parameters)

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            if self.strided_sample:
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:
                samples = importance[torch.randint(0, numel, (num_samples,), device=tensor.device)]

        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        if numel > num_samples:
            # code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/compressor/dgc.py
            for _ in range(self.max_adaptation_iters):
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()

        indices = indices[:num_selects]
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress
            if name != 'rnns.0.module.weight_hh_l0' and name != 'rnns.1.module.weight_hh_l0':
                tensor_compensated = self.memory.compensate(tensor, name, accumulate=True)
            else:
                tensor_compensated = tensor
            values, indices, numel, shape, num_selects = self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices,))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            values = torch.quantize_per_tensor(values, 0.01, 0, torch.qint8)
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            tensor = torch.quantize_per_tensor(tensor, 0.01, 0, torch.qint8)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.dequantize()
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)
