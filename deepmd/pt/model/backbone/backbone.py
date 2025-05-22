# SPDX-License-Identifier: LGPL-3.0-or-later
import torch 
import torch_npu 
# from torch_npu.contrib import transfer_to_npu


class BackBone(torch.nn.Module):
    def __init__(self, **kwargs):
        """BackBone base method."""
        super().__init__()

    def forward(self, **kwargs):
        """Calculate backBone."""
        raise NotImplementedError
