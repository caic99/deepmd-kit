# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation and
communication of the atomic properties according to output variable
definition `deepmd.dpmodel.OutputVariableDef`.

All models should be inherited from :class:`deepmd.pd.model.model.model.BaseModel`.
Models generated by `make_model` have already done it.
"""

import copy
import json

import numpy as np

from deepmd.pd.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pd.model.task import (
    BaseFitting,
)

from .dp_model import (
    DPModelCommon,
)
from .ener_model import (
    EnergyModel,
)
from .frozen import (
    FrozenModel,
)
from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)


def _get_standard_model_components(model_params, ntypes):
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    model_params["descriptor"]["type_map"] = copy.deepcopy(model_params["type_map"])
    descriptor = BaseDescriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", {})
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["type_map"] = copy.deepcopy(model_params["type_map"])
    fitting_net["mixed_types"] = descriptor.mixed_types()
    if fitting_net["type"] in ["dipole", "polar"]:
        fitting_net["embedding_width"] = descriptor.get_dim_emb()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = BaseFitting(**fitting_net)
    return descriptor, fitting, fitting_net["type"]


def _can_be_converted_to_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        # return false for any failure...
        return False


def _convert_preset_out_bias_to_array(preset_out_bias, type_map):
    if preset_out_bias is not None:
        for kk in preset_out_bias:
            if len(preset_out_bias[kk]) != len(type_map):
                raise ValueError(
                    "length of the preset_out_bias should be the same as the type_map"
                )
            for jj in range(len(preset_out_bias[kk])):
                if preset_out_bias[kk][jj] is not None:
                    if isinstance(preset_out_bias[kk][jj], list):
                        bb = preset_out_bias[kk][jj]
                    elif _can_be_converted_to_float(preset_out_bias[kk][jj]):
                        bb = [float(preset_out_bias[kk][jj])]
                    else:
                        raise ValueError(
                            f"unsupported type/value of the {jj}th element of "
                            f"preset_out_bias['{kk}'] "
                            f"{type(preset_out_bias[kk][jj])}"
                        )
                    preset_out_bias[kk][jj] = np.array(bb)
    return preset_out_bias


def get_standard_model(model_params):
    model_params_old = model_params
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    descriptor, fitting, fitting_net_type = _get_standard_model_components(
        model_params, ntypes
    )
    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])
    preset_out_bias = model_params.get("preset_out_bias")
    preset_out_bias = _convert_preset_out_bias_to_array(
        preset_out_bias, model_params["type_map"]
    )

    if fitting_net_type in ["ener", "direct_force_ener"]:
        modelcls = EnergyModel
    else:
        raise RuntimeError(f"Unknown fitting type: {fitting_net_type}")

    model = modelcls(
        descriptor=descriptor,
        fitting=fitting,
        type_map=model_params["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
        preset_out_bias=preset_out_bias,
    )
    model.model_def_script = json.dumps(model_params_old)
    return model


def get_model(model_params):
    model_type = model_params.get("type", "standard")
    if model_type == "standard":
        return get_standard_model(model_params)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(model_params)


__all__ = [
    "BaseModel",
    "DPModelCommon",
    "EnergyModel",
    "FrozenModel",
    "get_model",
    "make_model",
]
