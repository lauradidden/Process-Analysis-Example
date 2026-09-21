from logview.interfaces.predicate import Predicate
from logview.interfaces.query_evaluator import QueryEvaluator
from logview.interfaces.query_registry import QueryRegistry


class ResultSetCharacterizer:
    def get_properties(self, result_set, reference_log):
        raise NotImplementedError


class TwoResultSetsComparator:
    def get_properties(self, result_set_q, result_set_r, query_registry):
        raise NotImplementedError


class MultiResultSetsComparator:
    def get_properties(self, comparator_input):
        raise NotImplementedError
