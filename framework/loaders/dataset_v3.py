import numpy as np
import pandas as pd
import math
from common.utils import pprint
from tqdm import tqdm 
import gc
class Dataset(object):

    def __init__(self, feature, label, batch_size,
                 daily_batch=True, shuffle=False, pre_n_day=0):

        assert isinstance(feature, pd.DataFrame)
        assert isinstance(label, (pd.Series, pd.DataFrame))
        assert len(feature) == len(label) # ensure the same number of time stamps

        self.feature = feature
        label = label.reindex(feature.index)
        self.label = label
        # NOTE: always assume first index level is date
        self.count = label.groupby(level=0).size().values # [799,799,...,798,800] num of component stocks each day 

        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.daily_batch = daily_batch

        # calc size
        if self.daily_batch:
            self.indices = np.roll(np.cumsum(self.count), 1) # flatten then roll ahead 1 
            self.indices[0] = 0
            self.nb_batch = self.count // self.batch_size + (self.count % self.batch_size > 0) # batch_size : num of stocks in one batch
            # self.nb_batch -> [799,799,...,798,800] // batch_size
            self.length = self.nb_batch.sum()
            self.day_idx = self.label.index.get_level_values('datetime').unique()
        else:
            self.length = len(feature)

    @property
    def index(self):
        return self.feature.index

    def _iter_daily(self):
        indices = np.arange(len(self.nb_batch))# [0,1...,1936]
        if self.shuffle:
            np.random.shuffle(indices) # NOTE: only shuffle batches from different days
            pprint(f'day idx: {indices}') # !!! need to be set_random_seed in main
        for i in indices: # the ith daily batch
            for j in range(self.nb_batch[i]): # j_th batch in a day
                size = min(self.count[i] - j*self.batch_size, self.batch_size)
                start = self.indices[i] + j*self.batch_size
                yield (i,self.feature.iloc[start:start+size].values,
                       self.label.iloc[start:start+size].values)

    def _iter_daily_v2(self):
        assert self.batch_size == 800
        indices = np.arange(len(self.nb_batch))
        if self.shuffle:
            np.random.shuffle(indices) # NOTE: only shuffle batches from different days
            pprint(f'day idx: {indices}') # !!! need to be set_random_seed in main
        for i in indices: # the ith daily batch
            yield (i,self.feature.xs(self.day_idx[i],level='datetime').values,
                       self.label.xs(self.day_idx[i],level='datetime').values)

    def _iter_random(self):
        indices = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices[::self.batch_size]:
            yield (self.feature.iloc[i:i+self.batch_size].values,
                   self.label.iloc[i:i+self.batch_size].values)

    def __iter__(self):
        if self.daily_batch:
            return self._iter_daily()
        return self._iter_random()

    def __len__(self):
        return self.length

    def __add__(self, other):
        feature = pd.concat([self.feature, other.feature], axis=0)
        label = pd.concat([self.label, other.label], axis=0)
        return Dataset(feature, label, self.batch_size,
                       self.daily_batch, self.shuffle)

