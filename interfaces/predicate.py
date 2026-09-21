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


class Predicate(ABC):
    """
    Definition of a generic predicate. In LogView, any Predicate implementation is a component taking a log as an input
    argument and returning a log as an output result.
    """
    @abstractmethod
    def evaluate(self, log: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate the predicate's logic with the given log.
        :param log: A log as a pandas DataFrame.
        :return: a pandas DataFrame
        """

    @abstractmethod
    def as_string(self) -> str:
        """
        A human-readable representation of the predicate.
        :return: a string
        """
