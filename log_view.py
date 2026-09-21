"""
    This file is part of LogView.

    LogView is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LogView is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LogView. If not, see <https://www.gnu.org/licenses/>.
"""

from logview.interfaces import QueryEvaluator, QueryRegistry
from logview.interfaces import ResultSetCharacterizer, TwoResultSetsComparator, MultiResultSetsComparator
from logview.predicate.query import Query
from typing import Union, Set
from tabulate import tabulate

import warnings
import pandas as pd
import copy


class LogView:

    def __init__(self, query_evaluator: QueryEvaluator, query_registry: QueryRegistry,
                 initial_source_log: pd.DataFrame):
        if query_evaluator is None:
            raise RuntimeError("No query evaluator was defined in QueryRegistry!")
        if query_registry is None:
            raise RuntimeError("No query registry was defined in QueryRegistry!")

        initial_source_log.name = 'initial_source_log'
        # caches
        self.query_cache = {}
        self.result_set_name_cache = {'initial_source_log': initial_source_log}

        self.query_evaluator = query_evaluator
        self.query_registry = query_registry
        self.query_registry.set_initial_source_log(initial_source_log)

        # plugin registry
        self.result_set_characterizers = {}
        self.two_result_sets_comparators = {}
        self.multi_result_sets_comparators = {}

    def attach_result_set_characterizer(self, name: str, result_set_characterizer: ResultSetCharacterizer):
        self.result_set_characterizers[name] = result_set_characterizer

    def attach_two_result_sets_comparator(self, name: str, two_result_sets_comparator: TwoResultSetsComparator):
        self.two_result_sets_comparators[name] = two_result_sets_comparator

    def attach_multiquery_comparator(self, name: str, multi_result_sets_comparator: MultiResultSetsComparator):
        self.multi_result_sets_comparators[name] = multi_result_sets_comparator

    def evaluate_query(self, result_set_name: str, source_log: pd.DataFrame, query: Query) -> (pd.DataFrame, pd.DataFrame):
        cache_id = (id(source_log), query.as_string())
        if cache_id in self.query_cache:
            evaluation = self.query_registry.get_evaluation(self.query_cache[cache_id])
            if evaluation['result_set'].name != result_set_name:
                msg = (f"Ignoring the new name '{result_set_name}' since you are getting back an already computed "
                       f"result set with name {evaluation['result_set'].name}")
                warnings.warn(msg)
            return evaluation['result_set'], evaluation['complement_result_set']
        else:
            result_set, complement_result_set = self.query_evaluator.evaluate(source_log, query)
            result_set.name = result_set_name
            complement_result_set.name = f'complement_{result_set_name}'
            evaluation = {'query': copy.copy(query),
                          'source_log': source_log,
                          'result_set': result_set,
                          'complement_result_set': complement_result_set}
            self.query_registry.register_evaluation(id(result_set), evaluation)
            self.query_cache[cache_id] = id(result_set)
            self.result_set_name_cache[result_set.name] = result_set
            self.result_set_name_cache[complement_result_set.name] = complement_result_set
            return result_set, complement_result_set

    def characterize_result_set_with_reference_log(self, result_set, reference_log, characterizer_name: Union[str, Set[str]] = '*'):
        characterizers_to_run = LogView._get_requested_components(characterizer_name, self.result_set_characterizers)
        result_set_objs = self._turn_into_dataframe([result_set, reference_log])
        return self._characterize_result_set_with_reference_log(result_set_objs[0], result_set_objs[1], characterizers_to_run)

    def _characterize_result_set_with_reference_log(self, result_set: pd.DataFrame, reference_log: pd.DataFrame,
                                                    characterizers_to_run: set[str]):
        result = {}
        for characterizer_to_run in characterizers_to_run:
            if characterizer_to_run not in self.result_set_characterizers.keys():
                raise RuntimeError("The provided characterize_name is not known!")

            characterizer = self.result_set_characterizers[characterizer_to_run]
            properties = characterizer.get_properties(result_set, reference_log)
            if properties is not None:
                result[characterizer_to_run] = properties

        result_set_id = id(result_set)
        if result_set_id in self.query_cache:
            self.query_registry.annotate_result_set_with_properties(result_set_id, result)
        return result

    def compare_two_result_sets(self, result_set_q, result_set_r, comparator_name: Union[str, Set[str]] = '*'):
        comparators_to_run = LogView._get_requested_components(comparator_name, self.two_result_sets_comparators)
        result_set_objs = self._turn_into_dataframe([result_set_q, result_set_r])
        return self._compare_two_result_sets(result_set_objs[0], result_set_objs[1], comparators_to_run)

    def _compare_two_result_sets(self, result_set_q: pd.DataFrame, result_set_r: pd.DataFrame, comparators_to_run: set[str]):
        result = {}
        for comparator_to_run in comparators_to_run:
            if comparator_to_run not in self.two_result_sets_comparators.keys():
                raise RuntimeError("The provided comparator is not known!")

            comparator = self.two_result_sets_comparators[comparator_to_run]
            properties = comparator.get_properties(result_set_q, result_set_r, self.query_registry)
            if properties is not None:
                result[comparator_to_run] = properties
        return result

    def compare_result_sets(self, requested_result_sets: list = [], comparator_name: Union[str, Set[str]] = '*'):
        if len(requested_result_sets) == 0:
            result_set_ids = self.query_registry.get_registered_result_set_ids()
            self._compare_multi_result_sets(result_set_ids, comparator_name)
        else:
            requested_result_sets_obj = self._turn_into_dataframe(requested_result_sets)
            result_set_ids = [id(result_set) for result_set in requested_result_sets_obj]
            self._compare_multi_result_sets(result_set_ids, comparator_name)

    def compare_result_sets_with_label(self, requested_label, comparator_name: Union[str, Set[str]] = '*'):
        all_registered_result_set_ids = self.query_registry.get_registered_result_set_ids()
        result_set_ids = [result_set_id for result_set_id in all_registered_result_set_ids
                          if requested_label in self.query_registry.get_result_set_labels(result_set_id)]
        self._compare_multi_result_sets(result_set_ids, comparator_name)

    def _compare_multi_result_sets(self, result_set_ids: [int], comparator_name: Union[str, Set[str]]):
        evaluations = [self.query_registry.get_evaluation(result_set_id) for result_set_id in result_set_ids]
        comparator_input = [(evaluation['query'], evaluation['result_set']) for evaluation in evaluations]

        comparators_to_run = LogView._get_requested_components(comparator_name, self.multi_result_sets_comparators)
        for comparator_to_run in comparators_to_run:
            if comparator_to_run not in self.multi_result_sets_comparators.keys():
                raise RuntimeError("The provided comparer_name is not known!")

            comparator = self.multi_result_sets_comparators[comparator_to_run]
            _ = comparator.get_properties(comparator_input)

    def label_result_set(self, result_set: pd.DataFrame, label: str):
        self.query_registry.annotate_result_set_with_label(id(result_set), label)

    def get_summary(self, verbose: bool = True):
        result = self.query_registry.summary()
        if verbose:
            print(tabulate(result['evaluations'], headers='keys', tablefmt='psql'))
            print(tabulate(result['queries'], headers='keys', tablefmt='psql'))
        return result

    def _turn_into_dataframe(self, any_form_result_sets: list):
        result = []
        for any_form_result_set in any_form_result_sets:
            if isinstance(any_form_result_set, str):
                result_set_obj = self.result_set_name_cache.get(any_form_result_set)
                if result_set_obj is None:
                    raise RuntimeError(f"The result set {any_form_result_set} is unknown!")
                result.append(result_set_obj)
            elif isinstance(any_form_result_set, pd.DataFrame):
                result.append(any_form_result_set)
            else:
                raise RuntimeError(f"The result_set {any_form_result_set} do not have a valid datatype!")
        assert len(result) == len(any_form_result_sets)
        return result

    @staticmethod
    def _get_requested_components(component_names: Union[str, Set[str]], available_components):
        if isinstance(component_names, str):
            if component_names == '*':
                return set(available_components.keys())
            else:
                return {component_names}
        else:
            return component_names

