#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   evoalgo.py contains methods for showing, saving and loading data 

"""

import numpy as np
import time
from curriculum_learning.specialist.manager import SpecialistManager
from data_interfaces.loaders.phantom import PhantomLoader
from data_interfaces.utils import set_root
set_root('evorobot-phantom')

class EvoAlgo(object):
    def __init__(self, env, policy, seed, fileini, filedir):
        self.env = env                       # the environment
        self.policy = policy                 # the policy
        self.seed = seed                     # the seed of the experiment
        self.fileini = fileini               # the name of the file with the hyperparameters
        self.filedir = filedir               # the directory used to save/load files
        self.bestfit = -999999999.0          # the fitness of the best agent so far
        self.bestsol = None                  # the genotype of the best agent so far
        self.bestgfit = -999999999.0         # the performance of the best post-evaluated agent so far
        self.bestgsol = None                 # the genotype of the best postevaluated agent so far
        self.stat = np.arange(0, dtype=np.float64) # a vector containing progress data across generations
        self.avgfit = 0.0                    # the average fitness of the population
        self.last_save_time = time.time()    # the last time in which data have been saved
        self.policy_trials = self.policy.ntrials

        self.phantom_interface = PhantomLoader(
            self.seed,
            self.policy_trials,
            self.__env_name
        )
        self.specialist_manager = SpecialistManager(
            'main',
            self.__env_name,
            self.seed
        )
        self.init_specialist()

        self.cgen = None
        self.test_limit_stop = None

    def init_specialist(self):
        config = dict(
            fit_batch_size=50,
            score_batch_size=50,
            start_generation=1000,
            generation_trials=self.policy_trials
        )
        self.specialist_manager.add_specialist('main', config)

    @property
    def __env_name(self):
        return self.fileini.split('/')[2].split('/')[0]

    @property
    def end_generation(self):
        return self.phantom_interface.max_gen

    @property
    def progress(self):
        return (self.cgen / self.end_generation) * 100

    @property
    def cgen(self):
        return self._cgen

    @cgen.setter
    def cgen(self, cgen):
        self._cgen = cgen
        self.specialist_manager.generation = cgen

    @property
    def evaluation_seed(self):
        return self.seed + (self.cgen * self.batchSize)

    def save_all(self):
        self.specialist_manager.save()

    def process_conditions(self):
        data = np.array(self.conditions_data)
        return [list(r) for r in data]

    def process_specialist(self):
        gen_data = self.process_conditions()
        self.specialist_manager.update_data(gen_data)
        self.specialist_manager.process_generation()
        self.specialist_manager.save_stg()

    def process_integrations(self):
        self.process_conditions()
        self.process_specialist()

    def read_gen_data(self):
        self.steps, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter = self.phantom_interface.read_evolution(self.cgen)
        self.conditions_data = self.phantom_interface.read_conditions(self.cgen)

    def reset(self):
        self.bestfit = -999999999.0
        self.bestsol = None
        self.bestgfit = -999999999.0
        self.bestgsol = None
        self.stat = np.arange(0, dtype=np.float64)
        self.avgfit = 0.0
        self.last_save_time = time.time()

    def run(self, nevals):
        # Run method depends on the algorithm
        raise NotImplementedError
