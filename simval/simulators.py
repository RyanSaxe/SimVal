import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Callable
import inspect

class DataSimulator:
    def __init__(
        self,
        instructions: dict = None,
        interactions: dict = None,
    ):
        if instructions is None:
            instructions = {
                'simulation': dict()
            }
        defaults = self._get_defaults()
        #these are not in `get_defaults` since they are required. If `_get_defaults`
        # is overwritten, this makes sure distribution and priority_q are handled
        defaults.setdefault('distribution',np.random.normal)
        defaults.setdefault('priority_q',[])
        self.interactions = interactions
        self.instructions = {name: {**defaults, **rules} for name,rules in instructions.items()}

    def _get_defaults(self) -> dict:
        #overwrite this to create more involved default functions
        return dict()

    def simulate(self, size: int) -> pd.DataFrame:
        df = self._create_base(size)
        return self._apply_interactions(df)
            
    def _create_base(self, size: int) -> pd.DataFrame:
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

    def _apply_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, rules in self.interactions.items():
            df[column] = None
            preprocessed = ['priority_q']
            for name in rules['priority_q']:
                interaction = rules[name]
                df[column] = self._create_interaction(interaction, df)
                preprocessed.append(name)
            for name, interaction in self.interactions.items():
                if name in preprocessed:
                    continue
                df[column] = self._create_interaction(interaction, df)
        return df

    def _create_interaction(self, interaction: dict, df: pd.DataFrame):
        function = interaction['f']
        cols = interaction['cols']
        kwargs = interaction.get('kwargs',dict())
        split = interaction.get('split',False)
        axis = interaction.get('axis',None)
        if axis is None:
            output = function(
                *[df[c] for c in cols] if split else df[cols],
                **kwargs
            )
        else:
            output = df[cols].apply(lambda x: function(
                    *x if split else x,
                    **kwargs
                ), axis=axis
            )
        return output

