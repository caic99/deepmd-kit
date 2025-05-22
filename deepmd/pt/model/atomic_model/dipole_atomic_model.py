# SPDX-License-Identifier: LGPL-3.0-or-later

import torch 
import torch_npu 
# from torch_npu.contrib import transfer_to_npu

from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, DipoleFittingNet):
            raise TypeError(
                "fitting must be an instance of DipoleFittingNet for DPDipoleAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        # dipole not applying bias
        return ret
