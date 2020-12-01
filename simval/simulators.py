import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Callable
import inspect

class DataSimulator:
    def __init__(
        self,
        columns: dict = None,
    ):
        if columns is None:
            columns = {
                'simulation': dict()
            }
        self.instructions = {c:dict() for c in columns.keys()}
        defaults = self._get_defaults()
        assert 'distribution' in defaults, 'Must have a default manner of creating data distribution'
        for column,rules in columns.items():
            rules = {**defaults, **rules}
            self.instructions[column]['priority_q'] = rules.pop('priority_q',[])
            for name, build in rules.items():
                self._set_instructions(column, name, build)

    def _get_defaults(self) -> dict:
        raise NotImplementedError("Override Me! Don't forget to include 'distribution', as it is required")

    def _set_instructions(self, column: str, name: str, build: Union[tuple, list, Callable]) -> None:
        self.instructions[column][name] = dict()
        if not isinstance(build,(tuple, list)):
            assert callable(build), 'the function given is not callable'
            self.instructions[column][name]['f'] = build
            self.instructions[column][name]['kwargs'] = dict()
        else:
            assert len(build) == 2, 'build must be a tuple of (function, kwargs)'
            assert isinstance(build[1], dict), 'the second build element must be a dictionary'
            assert callable(build[0]), 'the function given is not callable'
            self.instructions[column][name]['f'] = build[0]
            self.instructions[column][name]['kwargs'] = build[1]
            
    def create_dataframe(self, size: int) -> pd.DataFrame:
        data_dict = dict()
        for column, rules in self.instructions.items():
            data = rules['distribution']['f'](size=size, **rules['distribution']['kwargs'])
            preprocessed = ['distribution', 'priority_q']
            for name in rules['priority_q']:
                instructions = rules[name]
                data = instructions['f'](data, **instructions['kwargs'])
                preprocessed.append(name)
            for name, instructions in rules.items():
                if name in preprocessed:
                    continue
                data = instructions['f'](data, **instructions['kwargs'])
            data_dict[column] = data
        return pd.DataFrame(data_dict)

class SimpleSalesSimulator(DataSimulator):
    def __init__(
        self,
        columns: dict = None,
        minimum: float = 100.0,
    ):
        super().__init__(columns = columns)
        self.minimum = minimum

    def _get_defaults(self) -> dict:
        return {
            'distribution': (
                self._distribution, {
                    'loc':0.0,
                    'scale':0.1
                }
            ),
            'flip_negative': np.absolute,
            'set_minimum': lambda x: (x + 1) * minimum,
            'priority_q': ['flip_negative','set_minimum']
        }

    def _distribution(self, size, loc, scale):
         return np.cumsum(np.random.normal(size=size, loc=loc, scale=scale))