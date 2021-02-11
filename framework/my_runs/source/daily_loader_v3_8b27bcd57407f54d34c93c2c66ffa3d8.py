import os
import numpy as np
import pandas as pd
import gc 
from common.utils import pprint, robust_zscore, download_http_resource
from loaders.dataset_v3 import Dataset
from tqdm import tqdm
import copy
from sacred import Ingredient
import pickle
DATA_SERVER = os.environ.get('DATA_SERVER', 'http://10.150.144.154:10020')


data_ingredient = Ingredient('daily_loader_v3')

EPS = 1e-12

@data_ingredient.config
def data_config():
    dset = 'day_csi300' # dataset
    label_id = 0 # LABEL$i
    log_transform = False

    train_start_date = '2007-01-01'
    train_end_date = '2014-12-31'
    valid_start_date = '2015-01-01'
    valid_end_date = '2016-12-31'
    test_start_date = '2017-01-01'
    test_end_date = '2019-06-18'

    batch_size = 800
    daily_batch = True
    pre_n_day = 1
    train_shuffle = True
    DATA_PATH = None
    
class DataLoader(object):

    @data_ingredient.capture
    def __init__(self, dset, label_id, batch_size,
                 daily_batch=True, log_transform=False, pre_n_day=10,train_shuffle=True, DATA_PATH=None):

        self.dset = dset
        self.label_id = label_id
        self.batch_size = batch_size
        self.daily_batch = daily_batch
        self.log_transform = log_transform
        self.train_shuffle = train_shuffle
        self.pre_n_day = pre_n_day
        self.DATA_PATH = DATA_PATH
        self._init_data()
        

    def _init_data(self):

        fname = os.path.join(self.DATA_PATH, self.dset+'.pkl')
        df = pd.read_pickle(fname)
        print(f'DATA_PATH : {fname}')
        
        # NOTE: ensure the multiindex like <datetime, instrument>
        if df.index.names[0] == 'instrument':
            df.index = df.index.swaplevel()
            df = df.sort_index()

        df['LABEL'] = df['LABEL%s'%self.label_id].groupby(level='datetime').apply(robust_zscore)
        # drop 'money' and 'pre_close' cols in hft_m_csi_800
        if self.dset[:3] != 'day':
            if df.columns.str.contains('money|pre_close').any():
                mask = ~df.columns.str.contains('money|pre_close')
                df = df.iloc[:,mask]
            if self.pre_n_day > 1:
                pprint('processing previous n day...')
                mask = ~df.columns.str.contains('LABEL|VWAP')
                feature = df.iloc[:,mask]
                label = df['LABEL']
                del df
                gc.collect()
                ins_box = []
                for ins in tqdm(feature.index.get_level_values('instrument').unique()):
                    _df = feature.xs(ins,level='instrument') #select the specific instrument df
                    _df = pd.concat([_df.shift(i) for i in reversed(range(self.pre_n_day))],axis=1)
                    #-> feature_1_pre_n_day...featrue_m_pre_n_day...feature_1_today...feature_m_today
                    _df['instrument'] = ins
                    ins_box.append(_df)
                feature = pd.concat(ins_box).reset_index()
                feature = feature.set_index(['datetime','instrument']).sort_index(level='datetime')
                label = label.reindex(feature.index)
                df = pd.concat([feature,label],axis=1)
                pprint(f'col {df.columns}')
                pprint(f'num_col {len(df.columns)}')
                pprint('finished !')
            self._raw_df = df

        elif self.dset[:3] == 'day':
            assert self.pre_n_day <= 60 #pre_n_day: daily_seq_len 
            col = []
            for i in ['OPEN', 'CLOSE', "HIGH", 'LOW', 'VOLUME', 'VWAP']:
                col.extend([i + str(j) for j in reversed(range(self.pre_n_day))]) # day_csi800 needs reserved feature cols for rnn
            _df = copy.deepcopy(df[col])
            _df['LABEL'] = df['LABEL'].values
            pprint(f'num_col {len(_df.columns)}')
            self._raw_df  = _df

    @data_ingredient.capture
    def load_data(self, train_start_date, train_end_date, valid_start_date,
                   valid_end_date, test_start_date, test_end_date):

        # string to datetime
        dates = (train_start_date, train_end_date, valid_start_date,
                 valid_end_date, test_start_date, test_end_date)
        (train_start_date, train_end_date, valid_start_date,
         valid_end_date, test_start_date, test_end_date) = [pd.Timestamp(x) for x in dates]

        # slice data
 
        train = self._raw_df.loc[train_start_date:train_end_date].dropna(subset=['LABEL'])
        valid = self._raw_df.loc[valid_start_date:valid_end_date].dropna(subset=['LABEL'])
        test  = self._raw_df.loc[test_start_date:test_end_date]

        pprint('train: {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(train), train_start_date, train_end_date))
        pprint('valid: {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(valid), valid_start_date, valid_end_date))
        pprint('test : {} samples, from {:%Y-%m-%d} to {:%Y-%m-%d}'.format(
            len(test), test_start_date, test_end_date))

        # preprocess
        pprint('Preprocess Start')
        mask = ~train.columns.str.contains('^LABEL|^WEIGHT|^GROUP')
        x_train = train.iloc[:, mask]
        y_train = train['LABEL']
        del train
        gc.collect()
        x_valid = valid.iloc[:, mask]
        y_valid = valid['LABEL']
        del valid
        gc.collect()
        x_test = test.iloc[:, mask]
        y_test = test['LABEL']
        del test
        gc.collect()
        if self.log_transform:
            x_train = np.log1p(np.abs(x_train))*np.sign(x_train)
            x_valid = np.log1p(np.abs(x_valid))*np.sign(x_valid)
            x_test = np.log1p(np.abs(x_test))*np.sign(x_test)

        mean = x_train.mean()
        std = x_train.std()
        # if self.dset[:3] != 'day':
            # x_train.to_pickle('/home/amax/data/x_train_ori.pkl')
            # pickle.dump(mean, open("mean_0707" + ".pkl", "wb"))
            # pickle.dump(std, open("std_0707" + ".pkl", "wb"))
        # NOTE: fillna(0) before zscore will cause std != 1
        x_train = (x_train - mean).div(std + EPS).fillna(0)
        x_valid = (x_valid - mean).div(std + EPS).fillna(0)
        x_test = (x_test - mean).div(std + EPS).fillna(0)

        train_set = Dataset(x_train, y_train, batch_size=self.batch_size,
                            daily_batch=self.daily_batch, shuffle=self.train_shuffle, pre_n_day=self.pre_n_day)
        valid_set = Dataset(x_valid, y_valid, batch_size=self.batch_size,
                            daily_batch=True, shuffle=False, pre_n_day=self.pre_n_day)
        test_set  = Dataset(x_test, y_test, batch_size=self.batch_size,
                            daily_batch=True, shuffle=False, pre_n_day=self.pre_n_day)

        return train_set, valid_set, test_set
