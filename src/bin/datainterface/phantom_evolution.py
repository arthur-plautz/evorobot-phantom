from operator import index
import pandas as pd
from utils import get_root_dir

class PhantomEvolution:
    def __init__(self, seed, trials, experiment) -> None:
        self.data_dir = f'{self.__root_dir}/data/{experiment}'
        self.seed = seed
        self.generation_trials = trials
        self.load()

    @property
    def __root_dir(self):
        return get_root_dir()

    def load(self):
        evolution_data_path = f'{self.data_dir}/runstats/s{self.seed}_run.csv'
        self.evolution_data = pd.read_csv(evolution_data_path, index_col='gen')

        initial_conditions_data_path = f'{self.data_dir}/initialconditions/s{self.seed}_run.csv'
        self.initial_conditions_data = pd.read_csv(initial_conditions_data_path, index_col='gen')

    @property
    def max_gen(self):
        return max(self.evolution_data.index)

    def read_conditions(self, gen):
        first_row = (gen-1) * self.generation_trials
        last_row = (gen * self.generation_trials)
        return self.initial_conditions_data.iloc[first_row:last_row]

    def read_evolution(self, gen):
        return list(self.evolution_data.iloc[gen-1])
