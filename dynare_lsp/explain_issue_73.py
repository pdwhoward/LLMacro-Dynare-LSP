"""Diagnostic documentation registered by GitHub issue #73."""

from .explain import _ENTRIES

_ENTRIES.update({
    "E070": {
        "title": "Incomplete deterministic shock schedule",
        "body": "A deterministic shock schedule supplies `periods` without `values`, or `values` without `periods`. Dynare requires both parts for a dated deterministic shock.",
    },
    "E071": {
        "title": "Deterministic shock periods and values do not conform",
        "body": "The literal `periods` specification expands to a different number of dates than the explicitly listed `values`. Make the dated values conform one-for-one, or use a scalar/vector expression with Dynare-compatible dimensions.",
    },
    "E072": {
        "title": "Invalid deterministic shock period range",
        "body": "A literal deterministic-shock period or range is non-positive or reversed. Deterministic simulation periods are positive dates and a literal range must run from its lower endpoint to its upper endpoint.",
    },
    "W114": {
        "title": "Deterministic shock period assigned more than once",
        "body": "The same deterministic shock has more than one statically visible assignment for the same period. A later assignment can override the earlier value, so inspect the overlapping `periods` schedules.",
    },
})
