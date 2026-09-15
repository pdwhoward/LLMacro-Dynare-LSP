"""Diagnostic documentation registered by GitHub issue #69."""

from .explain import _ENTRIES

_ENTRIES["W123"] = {
    "title": "Endogenous variable absent at the current period",
    "body": (
        "An endogenous variable appears in the dynamic model only at leads "
        "and/or lags after applying Dynare's `predetermined_variables` timing "
        "convention, with no effective time-t occurrence. Dynare's "
        "`model_diagnostics` reports this structural condition. The diagnostic "
        "does not infer which timing index is intended; inspect the model "
        "equations and declaration."
    ),
}
