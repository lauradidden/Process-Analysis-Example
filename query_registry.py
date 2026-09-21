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

from logview.interfaces import QueryRegistry
import pandas as pd


class QueryRegistryImpl(QueryRegistry):

    def __init__(self):
        self.registry = {}
        self.initial_source_log = None

    def set_initial_source_log(self, source_log: pd.DataFrame):
        if self.initial_source_log is not None:
            raise RuntimeError("Only an initial source log can be defined!")
        self.initial_source_log = source_log

    def get_initial_source_log(self):
        return self.initial_source_log

    def get_initial_source_log_id(self) -> int:
        return id(self.initial_source_log)

    def register_evaluation(self, result_set_id: int, evaluation: dict) -> None:
        for field in {'query', 'source_log', 'result_set', 'complement_result_set'}:
            if field not in evaluation:
                raise RuntimeError(f'The field {field} is not present in the evaluation!')

        if result_set_id in self.registry:
            raise RuntimeError(f"The result set id {result_set_id} cannot be registered twice!")

        self.registry[result_set_id] = {}
        self.registry[result_set_id]['evaluation'] = evaluation
        self.registry[result_set_id]['properties'] = []
        self.registry[result_set_id]['labels'] = []

    def get_registered_result_set_ids(self) -> list[int]:
        return list(self.registry.keys())

    def get_evaluation(self, result_set_id: int) -> dict:
        registry_item = self._get_registry_item(result_set_id)
        return registry_item['evaluation']

    def annotate_result_set_with_label(self, result_set_id: int, label: str):
        registry_item = self._get_registry_item(result_set_id)
        if label not in registry_item['labels']:
            registry_item['labels'].append(label)

    def get_result_set_labels(self, result_set_id: int) -> list[str]:
        registry_item = self._get_registry_item(result_set_id)
        return registry_item['labels']

    def annotate_result_set_with_properties(self, result_set_id: int, properties: dict) -> None:
        registry_item = self._get_registry_item(result_set_id)
        registry_item['properties'] = properties

    def get_result_set_properties(self, result_set_id: int) -> dict:
        registry_item = self._get_registry_item(result_set_id)
        return registry_item['properties']

    def summary(self) -> dict:
        registry_as_pd = {'source_log': [], 'query': [], 'result_set': [], 'labels': []}
        for key, value in self.registry.items():
            evaluation = value['evaluation']
            registry_as_pd['source_log'].append(evaluation['source_log'].name)
            registry_as_pd['query'].append(evaluation['query'].name)
            registry_as_pd['result_set'].append(evaluation['result_set'].name)
            registry_as_pd['labels'].append(value['labels'])
        evaluations = pd.DataFrame.from_dict(registry_as_pd)

        queries_as_pd = {'query': [], 'predicates': []}
        for key, value in self.registry.items():
            evaluation = value['evaluation']
            queries_as_pd['query'].append(evaluation['query'].get_name())
            queries_as_pd['predicates'].append(evaluation['query'].as_string())
        queries = pd.DataFrame.from_dict(queries_as_pd)

        return {'evaluations': evaluations, 'queries': queries}

    def _get_registry_item(self, result_set_id: int):
        registry_item = self.registry.get(result_set_id)
        if registry_item is None:
            raise RuntimeError("The provided result_set is not known!")
        return registry_item
