# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import math
from typing import (
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    get_hessian_name,
)


def make_hessian_model(T_Model):
    """Make a model that can compute Hessian.

    LIMITATION: this model is not jitable due to the restrictions of torch jit script.

    LIMITATION: only the hessian of `forward_common` is available.

    Parameters
    ----------
    T_Model
        The model. Should provide the `forward_common` and `atomic_output_def` methods

    Returns
    -------
    The model computes hessian.

    """

    class CM(T_Model):
        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            super().__init__(
                *args,
                **kwargs,
            )
            self.hess_fitting_def = copy.deepcopy(super().atomic_output_def())

        def requires_hessian(
            self,
            keys: Union[str, list[str]],
        ) -> None:
            """Set which output variable(s) requires hessian."""
            if isinstance(keys, str):
                keys = [keys]
            for kk in self.hess_fitting_def.keys():
                if kk in keys:
                    self.hess_fitting_def[kk].r_hessian = True

        def atomic_output_def(self):
            """Get the fitting output def."""
            return self.hess_fitting_def

        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> dict[str, torch.Tensor]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            vdef = self.atomic_output_def()
            hess_yes = [vdef[kk].r_hessian for kk in vdef.keys()]
            if any(hess_yes):
                coord.requires_grad = True
            # assert self.training # todo
            ret:dict[str,torch.Tensor] = super().forward_common(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )

            if any(hess_yes):
                if (
                    vdef["energy"].r_hessian
                    and sum(hess_yes) == 1
                    and "energy_derv_r" in ret
                    # and nf:=coord.shape[0] == 1 # large overhead for nf>1; maybe a warning is fine?
                ):  # use force to calculate energy hessian
                    force:torch.Tensor = ret["energy_derv_r"].squeeze(-2) # nf x nloc x 3
                    hess = torch.zeros(*force.shape, *coord.shape[-2:], device=force.device, dtype=force.dtype) # nf, nloc, 3, nloc, 3
                    # TODO: lazy compute
                    for nf in range(force.shape[0]):
                        for nloc in range(force.shape[1]):
                            for i in range(3):
                                # TODO: possibility of parallelization?
                                hess[nf, nloc, i] = torch.autograd.grad(
                                    outputs=force[nf, nloc, i],
                                    inputs=coord,
                                    create_graph=True,
                                )[0][nf] # only [nf] contains values, other chunks are zero

                    hess = hess.view(
                        force.shape[0], 1, force.shape[1] * 3, force.shape[1] * 3
                    )  # (nf, 1, nloc * 3, nloc * 3)
                    hess = {get_hessian_name("energy"): -hess}  # negative sign for force
                else:
                    hess = self._cal_hessian_all(
                        coord,
                        atype,
                        box=box,
                        fparam=fparam,
                        aparam=aparam,
                    )  # (nf, *vshape, nloc * 3, nloc * 3)
                ret.update(hess)
            return ret

        def _cal_hessian_all(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            nf, nloc = atype.shape
            coord = coord.view([nf, (nloc * 3)])
            box = box.view([nf, 9]) if box is not None else None
            fparam = fparam.view([nf, -1]) if fparam is not None else None
            aparam = aparam.view([nf, nloc, -1]) if aparam is not None else None
            fdef = self.atomic_output_def()
            # keys of values that require hessian
            hess_keys: list[str] = []
            for kk in fdef.keys():
                if fdef[kk].r_hessian:
                    hess_keys.append(kk)
            # result dict init by empty lists
            res = {get_hessian_name(kk): [] for kk in hess_keys}
            # loop over variable
            for kk in hess_keys:
                vdef = fdef[kk]
                vshape = vdef.shape
                vsize = math.prod(vdef.shape)
                # loop over frames
                for ii in range(nf):
                    icoord = coord[ii]
                    iatype = atype[ii]
                    ibox = box[ii] if box is not None else None
                    ifparam = fparam[ii] if fparam is not None else None
                    iaparam = aparam[ii] if aparam is not None else None
                    # loop over all components
                    for idx in range(vsize):
                        hess = self._cal_hessian_one_component(
                            idx, icoord, iatype, ibox, ifparam, iaparam
                        )
                        res[get_hessian_name(kk)].append(hess)
                res[get_hessian_name(kk)] = torch.stack(res[get_hessian_name(kk)]).view(
                    (nf, *vshape, nloc * 3, nloc * 3)
                )
            return res

        def _cal_hessian_one_component(
            self,
            ci,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # coord, # (nloc x 3)
            # atype, # nloc
            # box: Optional[torch.Tensor] = None,     # 9
            # fparam: Optional[torch.Tensor] = None,  # nfp
            # aparam: Optional[torch.Tensor] = None,  # (nloc x nap)
            wc = wrapper_class_forward_energy(self, ci, atype, box, fparam, aparam)
            hess = torch.autograd.functional.hessian(
                wc,
                coord,
                create_graph=self.training,
            )
            return hess

    class wrapper_class_forward_energy:
        def __init__(
            self,
            obj: CM,
            ci: int,
            atype: torch.Tensor,
            box: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor],
            aparam: Optional[torch.Tensor],
        ) -> None:
            self.atype, self.box, self.fparam, self.aparam = atype, box, fparam, aparam
            self.ci = ci
            self.obj = obj

        def __call__(
            self,
            coord,
        ) -> torch.Tensor:
            ci = self.ci
            atype, box, fparam, aparam = self.atype, self.box, self.fparam, self.aparam
            res = super(CM, self.obj).forward_common(
                coord.unsqueeze(0),
                atype.unsqueeze(0),
                box.unsqueeze(0) if box is not None else None,
                fparam.unsqueeze(0) if fparam is not None else None,
                aparam.unsqueeze(0) if aparam is not None else None,
                do_atomic_virial=False,
            )
            er = res["energy_redu"][0].view([-1])[ci]
            return er

    return CM
