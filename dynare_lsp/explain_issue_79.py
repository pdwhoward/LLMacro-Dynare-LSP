"""Diagnostic documentation registered by GitHub issue #79."""

from .explain import _ENTRIES

_ENTRIES["W098"] = {
    "title": "First-order observable innovation matrix is rank deficient",
    "body": (
        "The model has enough shocks/measurement errors by count, but after "
        "solving the determinate first-order linearization their contemporaneous "
        "impacts on the declared observables do not span all observable "
        "directions. This is a local first-order stochastic-singularity check; "
        "it is not a data-fit, posterior-mode, or Hessian diagnostic."
    ),
}
