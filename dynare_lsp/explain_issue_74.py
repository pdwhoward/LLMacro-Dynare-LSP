"""Diagnostic documentation registered by GitHub issue #74."""

from .explain import _ENTRIES

_ENTRIES.update({
    "W096": {
        "title": "Estimated parameter lies outside mathematical prior support",
        "body": "An `estimated_params` initial value, bound, or prior mean violates an objective support restriction of the selected prior/parameter type (for example a correlation outside [-1,1], a negative standard deviation, or a beta-prior mean outside (0,1)). This check does not judge whether the prior is economically appropriate.",
    },
    "W097": {
        "title": "Invalid prior hyperparameters",
        "body": "The prior's literal hyperparameters are mathematically invalid: a standard deviation/scale is non-positive or non-finite, or a beta mean/standard-deviation pair would imply non-positive beta shape parameters. Symbolic expressions that cannot be evaluated are left to Dynare.",
    },
})
