"""Structural model summary, mirroring Dynare's ``model_info`` command.

Classifies endogenous variables by their dynamic timing (static / predetermined
/ forward-looking / mixed), reports the state-space dimensions used by the
Blanchard-Kahn check, and — reusing the steady-state solver's Dulmage-Mendelsohn
decomposition — describes the recursive block structure of the static model.
This needs no solving; it is a static read of the parsed model.
"""

from __future__ import annotations

from typing import Any, Dict

from .parser import ParsedModel


def classify_variable_timing(model: ParsedModel) -> Dict[str, Dict[str, Any]]:
    """Classify each endogenous variable by its dynamic timing.

    Returns ``{name: {"class": <str>, "offsets": [sorted ints]}}`` where class
    is one of ``static`` / ``predetermined`` / ``forward_looking`` / ``mixed``.
    This is a purely *static* read of the model (no steady-state solve, no
    Blanchard-Kahn run), so it is available even for models that do not solve.
    """
    try:
        from .bk_check import _extract_variable_timing
        timing = _extract_variable_timing(model)
    except Exception:
        timing = {}

    result: Dict[str, Dict[str, Any]] = {}
    for var in model.endogenous:
        offsets = timing.get(var.name, set())
        has_lead = any(o > 0 for o in offsets)
        has_lag = any(o < 0 for o in offsets)
        if has_lead and has_lag:
            cls = "mixed"
        elif has_lead:
            cls = "forward_looking"
        elif has_lag:
            cls = "predetermined"
        else:
            cls = "static"
        result[var.name] = {
            "class": cls,
            "offsets": sorted(offsets) if offsets else [0],
        }
    return result


def compute_model_info(model: ParsedModel) -> Dict[str, Any]:
    """Return a JSON-friendly structural summary of *model*."""
    endo = [v.name for v in model.endogenous]
    exo = [v.name for v in model.exogenous]
    params = [v.name for v in model.parameters]
    n = len(endo)
    equations = model.dynamic_model_equations()

    info: Dict[str, Any] = {
        "n_endogenous": n,
        "endogenous": endo,
        "n_exogenous": len(exo),
        "exogenous": exo,
        "n_parameters": len(params),
        "parameters": params,
        "n_equations": len(equations),
    }

    # --- timing classification ---
    timing_by_var = classify_variable_timing(model)
    static, predetermined, forward, mixed = [], [], [], []
    for name in endo:
        cls = timing_by_var.get(name, {}).get("class", "static")
        if cls == "mixed":
            mixed.append(name)
        elif cls == "forward_looking":
            forward.append(name)
        elif cls == "predetermined":
            predetermined.append(name)
        else:
            static.append(name)

    info.update({
        "static": static,
        "predetermined": predetermined,
        "forward_looking": forward,
        "mixed": mixed,
        "n_static": len(static),
        "n_predetermined": len(predetermined),
        "n_forward_looking": len(forward),
        "n_mixed": len(mixed),
        # Blanchard-Kahn dimensions: state variables carry a lag, jumpers a lead.
        "n_state_variables": len(predetermined) + len(mixed),
        "n_jumpers": len(forward) + len(mixed),
    })

    # --- recursive block structure (Dulmage-Mendelsohn) ---
    info["blocks"] = _block_summary(model, endo, n)
    return info


def _block_summary(model: ParsedModel, endo, n: int):
    """Best-effort block-triangular structure of the static model."""
    if n == 0:
        return None
    try:
        import numpy as np

        from .solver import _block_structure, _preprocess_model

        equations, _locals = _preprocess_model(model, endo)
        if len(equations) != n:
            return None
        pattern = np.zeros((n, n), dtype=bool)
        for i, equation in enumerate(equations):
            for j in equation.var_indices:
                if 0 <= j < n:
                    pattern[i, j] = True
        structure = _block_structure(pattern, n, np)
        if structure is None:
            return None
        _eq_of_var, blocks = structure
        sizes = [len(b) for b in blocks]
        return {
            "n_blocks": len(blocks),
            "block_sizes": sizes,
            "largest_block": max(sizes) if sizes else 0,
            "recursive": len(blocks) > 1,
        }
    except Exception:
        return None
