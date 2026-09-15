"""Diagnostic documentation registered by GitHub issue #72."""

from .explain import _ENTRIES

_ENTRIES.update({
    "E066": {"title": "histval and endval are incompatible", "body": "The deterministic-simulation setup uses both `histval` and `endval`. Choose one initialization scheme; this is a structural Dynare rule, not an inference about the intended path."},
    "E067": {"title": "steady after histval overwrites historical initialization", "body": "A `steady` command follows `histval`. Dynare's deterministic-simulation initialization semantics do not permit this ordering because the steady-state operation replaces the historical conditions."},
    "E068": {"title": "histval assigns an undeclared symbol", "body": "A `histval` assignment targets a name that is not declared as an endogenous or exogenous symbol."},
    "E069": {"title": "histval contains a future offset", "body": "A `histval` entry uses a positive time offset. Historical values are supplied at period 0 and negative lags; positive leads are not historical initialization."},
    "W054": {"title": "histval targets a non-state endogenous variable", "body": "An endogenous variable is present in `histval`, but the dynamic model has no effective lag for that variable. The diagnostic reports this structural fact and does not claim the value is economically wrong."},
    "W055": {"title": "Required historical state value is missing", "body": "A state variable needs one or more negative-lag values that are not supplied by `histval`. Without `all_values_required`, Dynare defaults missing historical values to zero; with `all_values_required`, missing values are an error."},
})
