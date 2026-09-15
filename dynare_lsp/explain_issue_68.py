"""Diagnostic documentation registered by GitHub issue #68."""

from .explain import _ENTRIES

_ENTRIES.update({
    "W072": {
        "title": "Singular 0/0 generalized eigenvalue in the BK/QZ pencil",
        "body": (
            "The generalized Schur decomposition contains an alpha/beta pair "
            "whose numerator and denominator are both below Dynare's "
            "`qz_zero_threshold` (default 1e-6). Dynare treats this as a "
            "singular 0/0 generalized eigenvalue, so the local linearized "
            "model does not have a unique solution. This is a numerical/model-"
            "structure verdict; it does not identify which timing choice was intended."
        ),
    },
    "I072": {
        "title": "BK eigenvalue lies near the QZ stability boundary",
        "body": (
            "A finite generalized eigenvalue lies very close to the current "
            "QZ stability cutoff (`1 + unit_root_tol`). Small numerical or "
            "parameter changes can change the stable/unstable classification, "
            "so interpret the BK verdict as numerically sensitive."
        ),
    },
})
