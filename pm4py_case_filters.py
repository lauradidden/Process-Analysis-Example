"""PM4Py case-preserving filters exposed as LogView predicates.

This module intentionally wraps only PM4Py filters that retain/drop complete
cases. Event-level, prefix/suffix/between, and OCEL filters are not wrapped
because they can cut cases into fragments or operate on object-centric logs.
"""
from __future__ import annotations

from typing import Any, Collection, Iterable, Sequence, Tuple, Union

import pandas as pd
import pm4py

from logview.interfaces import Predicate


ValueCollection = Union[Collection[Any], Any]
CASE_ID_COL = "case:concept:name"


UNSUPPORTED_PM4PY_FILTERS = {
    "filter_between",
    "filter_prefixes",
    "filter_suffixes",
}


def _as_list(values: ValueCollection) -> list:
    if isinstance(values, (set, tuple, list)):
        return list(values)
    return [values]


def _format_values(values: Iterable[Any]) -> str:
    formatted = [f"'{v}'" if isinstance(v, str) else str(v) for v in values]
    return ", ".join(sorted(formatted))


def _copy_name_and_attrs(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    if hasattr(source, "attrs"):
        target.attrs = source.attrs.copy()
    if hasattr(source, "name"):
        target.name = getattr(source, "name")
    return target


def _validate_case_preserving(source_log: pd.DataFrame, filtered_log: pd.DataFrame, predicate_name: str) -> None:
    """Raise if a selected case contains fewer rows than in the source log."""
    if not isinstance(source_log, pd.DataFrame) or not isinstance(filtered_log, pd.DataFrame):
        return
    if CASE_ID_COL not in source_log.columns or CASE_ID_COL not in filtered_log.columns:
        return

    source_sizes = source_log.groupby(CASE_ID_COL, sort=False).size()
    filtered_sizes = filtered_log.groupby(CASE_ID_COL, sort=False).size()
    split_cases = [case_id for case_id, size in filtered_sizes.items() if source_sizes.get(case_id, None) != size]
    if split_cases:
        preview = ", ".join(map(str, split_cases[:5]))
        raise RuntimeError(
            f"{predicate_name} returned partial cases ({preview}). "
            "This lightweight LogView wrapper only supports case-preserving filters."
        )


class PM4PyCaseFilter(Predicate):
    """Generic predicate wrapper around a case-preserving PM4Py filter function."""

    def __init__(self, function_name: str, *args, **kwargs):
        if function_name.startswith("filter_ocel"):
            raise ValueError("OCEL filters are intentionally not supported by this lightweight LogView wrapper.")
        if function_name in UNSUPPORTED_PM4PY_FILTERS:
            raise ValueError(f"{function_name} can split cases and is intentionally not supported.")
        if function_name == "filter_time_range" and kwargs.get("mode") == "events":
            raise ValueError("filter_time_range(mode='events') can split cases. Use 'traces_contained' or 'traces_intersecting'.")
        if function_name == "filter_event_attribute_values" and kwargs.get("level") == "event":
            raise ValueError("filter_event_attribute_values(level='event') can split cases. Use level='case'.")
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        fn = getattr(pm4py, self.function_name)
        result = fn(log, *self.args, **self.kwargs)
        _validate_case_preserving(log, result, self.__class__.__name__)
        return _copy_name_and_attrs(log, result)

    # Compatibility with evaluators that use apply() instead of evaluate().
    def apply(self, log: pd.DataFrame) -> pd.DataFrame:
        return self.evaluate(log)

    def as_string(self) -> str:
        args = ", ".join([repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()])
        return f"{self.function_name}({args})"


class EventAttributeValues(PM4PyCaseFilter):
    def __init__(self, attribute_key: str, values: ValueCollection, retain: bool = True):
        self.attribute_key = attribute_key
        self.values = _as_list(values)
        self.retain = retain
        super().__init__("filter_event_attribute_values", attribute_key, self.values, level="case", retain=retain)

    def as_string(self) -> str:
        op = "in" if self.retain else "not in"
        return f"({self.attribute_key} {op} {{ {_format_values(self.values)} }})"


class TraceAttributeValues(PM4PyCaseFilter):
    def __init__(self, attribute_key: str, values: ValueCollection, retain: bool = True):
        self.attribute_key = attribute_key
        self.values = _as_list(values)
        self.retain = retain
        super().__init__("filter_trace_attribute_values", attribute_key, self.values, retain=retain)

    def as_string(self) -> str:
        op = "in" if self.retain else "not in"
        return f"(TraceAttribute {self.attribute_key} {op} {{ {_format_values(self.values)} }})"


class Variants(PM4PyCaseFilter):
    def __init__(self, variants: Collection[Sequence[str]], retain: bool = True):
        self.variants = list(variants)
        self.retain = retain
        super().__init__("filter_variants", self.variants, retain=retain)

    def as_string(self) -> str:
        op = "in" if self.retain else "not in"
        variants_as_string = ", ".join(["<" + ", ".join(map(str, v)) + ">" for v in self.variants])
        return f"(Variants {op} {{ {variants_as_string} }})"


class DirectlyFollowsRelation(PM4PyCaseFilter):
    def __init__(self, relations: Collection[Tuple[str, str]], retain: bool = True):
        self.relations = list(relations)
        self.retain = retain
        super().__init__("filter_directly_follows_relation", self.relations, retain=retain)

    def as_string(self) -> str:
        op = "contains" if self.retain else "does not contain"
        rels = ", ".join([f"({a} -> {b})" for a, b in self.relations])
        return f"(DirectlyFollows {op} {{ {rels} }})"


class EventuallyFollowsRelation(PM4PyCaseFilter):
    def __init__(self, relations: Collection[Tuple[str, str]], retain: bool = True):
        self.relations = list(relations)
        self.retain = retain
        super().__init__("filter_eventually_follows_relation", self.relations, retain=retain)

    def as_string(self) -> str:
        op = "contains" if self.retain else "does not contain"
        rels = ", ".join([f"({a} => {b})" for a, b in self.relations])
        return f"(EventuallyFollows {op} {{ {rels} }})"


class TimeRange(PM4PyCaseFilter):
    def __init__(self, dt1: str, dt2: str, mode: str = "traces_intersecting"):
        if mode == "events":
            raise ValueError("filter_time_range(mode='events') can split cases. Use 'traces_contained' or 'traces_intersecting'.")
        self.dt1 = dt1
        self.dt2 = dt2
        self.mode = mode
        super().__init__("filter_time_range", dt1, dt2, mode=mode)

    def as_string(self) -> str:
        return f"(TimeRange {self.mode} [{self.dt1}, {self.dt2}])"


class CaseSize(PM4PyCaseFilter):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__("filter_case_size", min_size, max_size)

    def as_string(self) -> str:
        return f"(CaseSize [{self.min_size}, {self.max_size}])"


class CasePerformance(PM4PyCaseFilter):
    def __init__(self, min_performance: float, max_performance: float):
        self.min_performance = min_performance
        self.max_performance = max_performance
        super().__init__("filter_case_performance", min_performance, max_performance)

    def as_string(self) -> str:
        return f"(CasePerformance [{self.min_performance}, {self.max_performance}])"


class ActivitiesRework(PM4PyCaseFilter):
    def __init__(self, activity: str, min_occurrences: int = 2):
        self.activity = activity
        self.min_occurrences = min_occurrences
        super().__init__("filter_activities_rework", activity, min_occurrences=min_occurrences)

    def as_string(self) -> str:
        return f"(ActivitiesRework '{self.activity}' >= {self.min_occurrences})"


class PathsPerformance(PM4PyCaseFilter):
    def __init__(self, path: Tuple[str, str], min_performance: float, max_performance: float, keep: bool = True):
        self.path = tuple(path)
        self.min_performance = min_performance
        self.max_performance = max_performance
        self.keep = keep
        super().__init__("filter_paths_performance", self.path, min_performance, max_performance, keep=keep)

    def as_string(self) -> str:
        op = "keeps" if self.keep else "removes"
        return f"(PathsPerformance {self.path[0]} -> {self.path[1]} {op} [{self.min_performance}, {self.max_performance}])"


class VariantsTopK(PM4PyCaseFilter):
    def __init__(self, k: int):
        self.k = k
        super().__init__("filter_variants_top_k", k)

    def as_string(self) -> str:
        return f"(VariantsTopK {self.k})"