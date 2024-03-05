"""
positional encoding



"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder:
    "adapted from https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py#L48"
    def __init__(self, **kwargs):
        """
        default config:

        :param kwargs:
        """
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """

        :param inputs: (N_rays, N_samples, 3)
        :return: (N_rays, N_samples, D)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def test():
    ""
    # x = torch.randn(10, 50, 3)
    # embed, _ = get_embedder(10)
    # enc = embed(x)
    # print(enc.shape) # torch.Size([10, 50, 63])
    # print(x[0, :2])
    # print(enc[0, :2]) # this encoding already includes the input coordinates

    embed, _ = get_embedder(15, input_dims=1)
    enc = embed(torch.randn(1, 1, 1))
    print(enc.shape) # (1, 1, 31) 2*multires + input_dims

if __name__ == '__main__':
    test()