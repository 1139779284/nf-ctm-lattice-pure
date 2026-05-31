from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .neuron_field import LatticeNFCTMNeuronField
from .multiscale_field import CoupledMultiScaleLatticeCTMTrace, CoupledMultiScaleLatticeNFCTM


@dataclass
class LatticeHookHandle:
    handle: Any
    module: nn.Module
    layer: LatticeNFCTMNeuronField

    def remove(self) -> None:
        self.handle.remove()


def attach_lattice_nf_ctm_hook(module: nn.Module, layer: LatticeNFCTMNeuronField) -> LatticeHookHandle:
    """Attach NF-CTM Lattice as an in-flow feature replacement hook.

    This is intentionally tiny: it does not decode YOLO predictions, does not run
    any guard, and does not perform post-hoc selection.  The hook simply replaces
    the selected feature tensor with the terminal CTM neuron-field state.
    """

    def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output) and output.ndim == 4 and output.shape[1] == layer.channels:
            return layer(output)
        return output

    h = module.register_forward_hook(_hook)
    return LatticeHookHandle(handle=h, module=module, layer=layer)


def forward_with_coupled_lattice_nf_ctm(
    inner: nn.Module,
    modules: Mapping[Any, nn.Module],
    coupled: CoupledMultiScaleLatticeNFCTM,
    x: torch.Tensor,
    *,
    return_trace: bool = False,
) -> tuple[Any, CoupledMultiScaleLatticeCTMTrace | None, dict[str, torch.Tensor]]:
    """Run a frozen detector with one coupled multi-scale CTM field.

    The detector is executed once to collect its native feature maps, the single
    coupled CTM field advances all scales jointly, then a second detector pass
    receives the terminal CTM states at the selected native layers.  This avoids
    the v1 sandwich pattern where several independent CTM hooks are trained as
    separate plugins.
    """

    module_map = {str(k): v for k, v in modules.items()}
    refs: dict[str, torch.Tensor] = {}
    capture_handles = []

    def make_capture(key: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if torch.is_tensor(output) and output.ndim == 4:
                refs[key] = output.detach()
            return output

        return _hook

    try:
        for key, module in module_map.items():
            capture_handles.append(module.register_forward_hook(make_capture(key)))
        with torch.no_grad():
            inner(x)
    finally:
        for h in capture_handles:
            h.remove()

    missing = [k for k in coupled.scale_keys if k not in refs]
    if missing:
        raise ValueError(f"failed to capture coupled CTM feature scales: {missing}")

    ctm_out = coupled(refs, return_trace=return_trace)
    if isinstance(ctm_out, CoupledMultiScaleLatticeCTMTrace):
        terminals = ctm_out.final
        trace = ctm_out
    else:
        terminals = ctm_out
        trace = None

    replace_handles = []

    def make_replace(key: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if torch.is_tensor(output) and output.ndim == 4:
                return terminals[key].to(device=output.device, dtype=output.dtype)
            return output

        return _hook

    try:
        for key, module in module_map.items():
            replace_handles.append(module.register_forward_hook(make_replace(key)))
        raw = inner(x)
    finally:
        for h in replace_handles:
            h.remove()
    return raw, trace, refs
