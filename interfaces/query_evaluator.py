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
from logview.interfaces import Predicate
import pandas as pd


class QueryEvaluator(ABC):

    """
    Definition of a generic predicate evaluator. In LogView, any QueryEvaluator implementation is a component taking a
    log and a query as input arguments and returning the result set of the query with its complement as an output result.
    """
    @abstractmethod
    def evaluate(self, log: pd.DataFrame, query: Predicate) -> (pd.DataFrame, pd.DataFrame):
        """
        Evaluate a given query on a log.
        :param log: A log as a pandas DataFrame.
        :param query: A query as a predicate.
        :return: two pandas DataFrames, where the first is the result set of the query and second is its complement.
        """
