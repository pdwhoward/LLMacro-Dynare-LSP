"""Diagnostic documentation registered by GitHub issue #71."""

from .explain import _ENTRIES

_ENTRIES["W113"] = {
    "title": "Shock covariance matrix is not positive semidefinite",
    "body": (
        "The individual shock variances/correlations are syntactically valid, "
        "but their jointly implied variance-covariance matrix is not positive "
        "semidefinite. Pairwise correlations can all lie inside [-1, 1] and "
        "still be mutually inconsistent. Inspect the `var` and `corr` entries "
        "named by the diagnostic evidence."
    ),
}
