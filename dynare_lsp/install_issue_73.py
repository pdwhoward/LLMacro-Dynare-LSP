"""Install deterministic-shock checks into the existing shock diagnostic hook."""

from __future__ import annotations


def install() -> None:
    from . import shocks_diagnostics
    from .deterministic_shock_diagnostics import check_deterministic_shocks

    if getattr(shocks_diagnostics, "_deterministic_schedule_extension_installed", False):
        return
    original = shocks_diagnostics.check_shocks

    def check_shocks(model, include_models=None, param_known=None):
        include_models = include_models or []
        diagnostics = original(model, include_models=include_models, param_known=param_known)
        diagnostics.extend(check_deterministic_shocks([model, *include_models]))
        return diagnostics

    shocks_diagnostics.check_shocks = check_shocks
    setattr(shocks_diagnostics, "_deterministic_schedule_extension_installed", True)
