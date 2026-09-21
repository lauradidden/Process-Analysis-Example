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

from abc import ABC, abstractmethod
import pandas as pd


class QueryRegistry(ABC):

    @abstractmethod
    def set_initial_source_log(self, source_log: pd.DataFrame):
        """
        Define the initial source log.
        :param source_log: a log as a pandas DataFrame.
        """

    @abstractmethod
    def get_initial_source_log(self):
        """
        Return the initial source log.
        :return: a pandas DataFrame.
        """

    @abstractmethod
    def get_initial_source_log_id(self) -> int:
        """
        Return the identifier of the initial source log.
        :return: an integer.
        """

    @abstractmethod
    def register_evaluation(self, result_set_id: int, evaluation: dict) -> None:
        """
        Register the evaluation of a result set.
        :param result_set_id: the identifier of a result set.
        :param evaluation: the evaluation of the result set as a dictionary.
        """

    @abstractmethod
    def get_registered_result_set_ids(self) -> list[int]:
        """
        Return the identifier of all registered result sets.
        :return: a list of integers.
        """

    @abstractmethod
    def get_evaluation(self, result_set_id: int) -> dict:
        """
        Return the evaluation of a result set.
        :return: a dictionary.
        """

    @abstractmethod
    def annotate_result_set_with_label(self, result_set_id: int, label: str):
        """
        Label a result set with a given label.
        :param result_set_id: the identifier of a result set.
        :param label: a label in string format.
        """

    @abstractmethod
    def get_result_set_labels(self, result_set_id: int) -> list[str]:
        """
        Return the labels of a result set.
        :param result_set_id: the identifier of a result set.
        :return: a list of strings.
        """

    @abstractmethod
    def annotate_result_set_with_properties(self, result_set_id: int, properties: dict) -> None:
        """
        Annotate a result set with a given set of properties.
        :param result_set_id: the identifier of a result set.
        :param properties: properties as a dictionary.
        """

    @abstractmethod
    def get_result_set_properties(self, result_set_id: int) -> dict:
        """
        Return the properties of a result set.
        :param result_set_id: the identifier of a result set.
        :return: a set of properties.
        """

    @abstractmethod
    def summary(self) -> dict:
        """
        Return a summary of query registry
        :return: a dictionary.
        """