# SPDX-License-Identifier: LGPL-3.0-or-later
import torch 
import torch_npu 
# from torch_npu.contrib import transfer_to_npu

from deepmd.dpmodel.descriptor import (
    make_base_descriptor,
)

BaseDescriptor = make_base_descriptor(torch.Tensor, "forward")
