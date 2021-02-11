# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

import loaders as loaders_module
import models as models_module

from common.utils import (pprint, set_random_seed, create_output_path,
                          zscore, robust_zscore, count_num_params)
from common.functions import K, rmse, mae  # BUG: sacred cannot save source files used in ingredients

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('hft_pred',
                ingredients=[
                    loaders_module.daily_loader_v3.data_ingredient,
                    models_module.cnn_rnn_v2.model_ingredient
                ])

create_output_path('my_runs')
ex.observers.append(
    FileStorageObserver("my_runs", "my_runs/resource", "my_runs/source",
                        "my_runs/templete"))

def Day_model_1(output_path, day_model_1, day_train_set, day_valid_set,
                day_test_set):
    pprint('training day_model_1 ...')
    train_hids, valid_hids = day_model_1.fit(day_train_set,
                                             day_valid_set,
                                             run=None)
    _, test_hids = day_model_1.predict(day_test_set)
    # file = open(output_path+'/train_hids.pkl','wb')
    # pickle.dump(train_hids,file)
    # file.close()

    # file = open(output_path+'/valid_hids.pkl','wb')
    # pickle.dump(valid_hids,file)
    # file.close()

    # file = open(output_path+'/test_hids.pkl','wb')
    # pickle.dump(test_hids,file)
    # file.close()

    # day_model_1.save(output_path + '/best_day_model.bin')

    # with open(output_path+'/train_hids.pkl', 'rb') as file:
    #     train_hids = pickle.load(file)
    #     file.close()
    # with open(output_path+'/valid_hids.pkl', 'rb') as file:
    #     valid_hids = pickle.load(file)
    #     file.close()
    # with open(output_path+'/test_hids.pkl', 'rb') as file:
    #     test_hids = pickle.load(file)
    #     file.close()

    # day_model_1.load(output_path + '/best_day_model.bin')
    pprint('inference...')
    pprint('validation set day_model_1 :')
    inference(dset=day_valid_set, model=day_model_1)
    pprint('testing set day_model_1 :')
    inference(dset=day_test_set, model=day_model_1)
    pprint('done.')

    return train_hids, valid_hids, test_hids

def Day_model_2(_run,
                output_path,
                day_model_2,
                day_train_set,
                day_valid_set,
                day_test_set,
                train_day_reps,
                valid_day_reps,
                test_day_reps,
                pred_path=None):
    pprint('training day_model_2')
    tune_train_reps, tune_valid_reps = day_model_2.fit(day_train_set,
                                                       day_valid_set,
                                                       train_day_reps,
                                                       valid_day_reps,
                                                       run=_run)

    pprint('inference...')
    pprint('validation set day_model_2:')
    inference(dset=day_valid_set,
              model=day_model_2,
              day_rep=valid_day_reps,
              pred_path=None)

    pprint('testing set day_model_2:')
    rmse, mae   = inference(dset=day_test_set,
                             model=day_model_2,
                             day_rep=test_day_reps,
                             pred_path=pred_path)
    return rmse, mae
    
def inference(dset, model,day_rep=None, pred_path=None):
    pred = pd.DataFrame(index=dset.index)
    pred['label'] = dset.label
    if day_rep is not None:
        pred['score'], _ = model.predict(dset, day_rep)
    else:
        pred['score'], _ = model.predict(dset)
    if pred_path is not None:
        pred.to_pickle(pred_path)
        ex.add_artifact(pred_path)

    robust_ic = pred.groupby(level='datetime').apply(
        lambda x: robust_zscore(x.label).corr(robust_zscore(x.score)))
    rank_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman'))
    pprint('Robust IC:', robust_ic.mean(), ',',
           robust_ic.mean() / robust_ic.std())
    pprint('Rank IC:', rank_ic.mean(), ',', rank_ic.mean() / rank_ic.std())
    ori_label = pd.read_pickle('/D_data/v-liuyan/data/%s.pkl'%day_loader.dset)['LABEL%s'%day_loader.label_id]
    pred['ori_label'] = ori_label.reindex(pred.index)
    _mean = pred['ori_label'].mean()
    _std = pred['ori_label'].std()
    pred.score = pred.score *  _std +  _mean
    RMSE = rmse(pred.score, pred.ori_label)
    MAE = mae(pred.score, pred.ori_label)
    pprint('RMSE:', RMSE)
    pprint('MAE:', MAE)

    return RMSE, MAE

def Min_model(_run,
              output_path,
              min_model,
              min_train_set,
              min_valid_set,
              min_test_set,
              train_hids,
              valid_hids,
              itera=0):

    pprint('training min_model...')
    train_day_reps, valid_day_reps = min_model.fit(min_train_set,
                                                   min_valid_set,
                                                   train_hids,
                                                   valid_hids,
                                                   output_path = output_path,
                                                   run=_run,
                                                   itera=itera)
    test_day_reps = min_model.predict(min_test_set)

    # file = open(output_path + '/train_day_reps_min_%d.pkl' % step, 'wb')
    # pickle.dump(train_day_reps, file)
    # file.close()

    # file = open(output_path + '/valid_day_reps_min_%d.pkl' % step, 'wb')
    # pickle.dump(valid_day_reps, file)
    # file.close()

    # file = open(output_path + '/test_day_reps_min_%d.pkl' % step, 'wb')
    # pickle.dump(test_day_reps, file)
    # file.close()

    # with open(output_path + '/train_day_reps_min_%d.pkl' % step, 'rb') as file:
    #     train_day_reps = pickle.load(file)
    #     file.close()
    # with open(output_path + '/valid_day_reps_min_%d.pkl' % step, 'rb') as file:
    #     valid_day_reps = pickle.load(file)
    #     file.close()
    # with open(output_path + '/test_day_reps_min_%d.pkl' % step, 'rb') as file:
    #     test_day_reps = pickle.load(file)
    #     file.close()
    return train_day_reps, valid_day_reps, test_day_reps

def Mix_model(_run,
              output_path,
              mix_model,
              min_train_set,
              min_valid_set,
              min_test_set,
              train_day_reps,
              valid_day_reps,
              itera=0):
    pprint('training mix_model...')
    train_day_reps, valid_day_reps = mix_model.fit(min_train_set,
                                                   min_valid_set,
                                                   train_day_reps,
                                                   valid_day_reps,
                                                   output_path = output_path,
                                                   run=_run,
                                                   itera=itera)
    test_day_reps = mix_model.predict(min_test_set)

    # file = open(output_path + '/train_day_reps_mix_%d.pkl' % step, 'wb')
    # pickle.dump(train_day_reps, file)
    # file.close()

    # file = open(output_path + '/valid_day_reps_mix_%d.pkl' % step, 'wb')
    # pickle.dump(valid_day_reps, file)
    # file.close()

    # file = open(output_path + '/test_day_reps_mix_%d.pkl' % step, 'wb')
    # pickle.dump(test_day_reps, file)
    # file.close()

    # with open(output_path + '/train_day_reps_mix_%d.pkl' % step, 'rb') as file:
    #     train_day_reps = pickle.load(file)
    #     file.close()
    # with open(output_path + '/valid_day_reps_mix_%d.pkl' % step, 'rb') as file:
    #     valid_day_reps = pickle.load(file)
    #     file.close()
    # with open(output_path + '/test_day_reps_mix_%d.pkl' % step, 'rb') as file:
    #     test_day_reps = pickle.load(file)
    #     file.close()
    return train_day_reps, valid_day_reps, test_day_reps

@ex.config
def run_config():
    seed = 2
    output_path = '/home/amax/Documents/HM_CNN_RNN/out'
    loader_name = 'daily_loader_v3'
    model_name = 'rnn_v3'
    comt = 'rnn_60_1.0'
    run_on = False
    dsets = ["day_csi800_ly_v3_1", "hft_10m_csi800_ly_v3_1"]


@ex.main
def main(_run, seed, loader_name, model_name, output_path, comt, run_on,
         dsets):
    # path
    output_path = create_output_path(output_path)

    pprint('output path:', output_path)
    model_path = output_path + '/model.bin'
    # pred_path = output_path+'/pred_%s_%d.pkl' %(model_name,seed)
    pprint('create loader `%s` and model `%s`...' % (loader_name, model_name))

    ###### Daily Model and Data Prepare #########
    set_random_seed(seed)
    global day_loader
    day_loader = getattr(loaders_module, loader_name).DataLoader(dset=dsets[0])
    day_model_1 = getattr(models_module, model_name).Day_Model_1()
    super_model = getattr(models_module, model_name)
    pprint(f'''
        Day_Model_1: {count_num_params(super_model.Day_Model_1())}, 
        Day_model_2: {count_num_params(super_model.Day_Model_2())}, Digger:{count_num_params(super_model.Min_Model())},
        Guider: {count_num_params(super_model.Mix_Model())}
        ''')
    pprint('load daily data...')
    day_train_set, day_valid_set, day_test_set = day_loader.load_data()

    ###### Day Model 1#######
    train_hids, valid_hids, test_hids = Day_model_1(output_path, day_model_1,
                                         day_train_set, day_valid_set,
                                         day_test_set)

    ###### High-freq Model and Data Prepare ########
    set_random_seed(seed)
    min_loader = getattr(loaders_module, loader_name).DataLoader(dset=dsets[1])

    pprint('load high-freq data...')
    min_train_set, min_valid_set, min_test_set = min_loader.load_data()

    ####### Min Model ######
    itera = 0
    set_random_seed(seed)
    _run = SummaryWriter(comment='_min_model_%d'%itera) if run_on else None
    min_model = getattr(models_module, model_name).Min_Model()
    min_train_day_reps, min_valid_day_reps, min_test_day_reps = Min_model(_run,
                                                              output_path,
                                                              min_model,
                                                              min_train_set,
                                                              min_valid_set,
                                                              min_test_set,
                                                              train_hids, 
                                                              valid_hids,
                                                              itera=itera)
    ######### Day Model 2########
    set_random_seed(seed)
    _run = SummaryWriter(comment='_day_model2_min_%d'%itera) if run_on else None
    day_model_2 = getattr(models_module, model_name).Day_Model_2()
    pred_path = output_path+'/pred_%s_%d.pkl' %(model_name,itera)
    rmse_min,  mae_min = Day_model_2(
        _run,
        output_path,
        day_model_2,
        day_train_set,
        day_valid_set,
        day_test_set,
        min_train_day_reps, 
        min_valid_day_reps, 
        min_test_day_reps,
        pred_path)
    
    #####Iter prep###
    pre_rmse_min = 100
    pre_mae_min = 100
    # rmse_mix = 0.0
    # mae_mix = 0.0
    #####Iter####
    while (rmse_min < pre_rmse_min) or (mae_min < pre_mae_min):
        itera += 1
        pprint('Iter:', itera)
        pre_rmse_min = rmse_min
        pre_mae_min = mae_min
        # pre_rmse_mix = rmse_mix
        # pre_mae_mix = mae_mix
        ####### Mix Model ######
        set_random_seed(seed)
        mix_model = getattr(models_module, model_name).Mix_Model()
        _run = SummaryWriter(comment='_mix_model_%d' %
                             itera) if run_on else None
        mix_train_day_reps, mix_valid_day_reps, mix_test_day_reps = Mix_model(
            _run,
            output_path,
            mix_model,
            min_train_set,
            min_valid_set,
            min_test_set,
            min_train_day_reps,
            min_valid_day_reps,
            itera=itera)

        ####### Day Model 2######
        set_random_seed(seed)
        pprint('Mix model fine tune...')
        day_model_2 = getattr(models_module, model_name).Day_Model_2()
        _run = SummaryWriter(comment='_day_model2_mix_%d' %
                             itera) if run_on else None
        rmse_mix, mae_mix = Day_model_2(
            _run,
            output_path,
            day_model_2,
            day_train_set,
            day_valid_set,
            day_test_set,
            mix_train_day_reps,
            mix_valid_day_reps,
            mix_test_day_reps)

        ####### Min Model ######
        set_random_seed(seed)
        min_model = getattr(models_module, model_name).Min_Model()
        _run = SummaryWriter(comment='_min_model_%d' %
                             itera) if run_on else None

        min_train_day_reps, min_valid_day_reps, min_test_day_reps = Min_model(
            _run,
            output_path,
            min_model,
            min_train_set,
            min_valid_set,
            min_test_set,
            mix_train_day_reps,
            mix_valid_day_reps,
            itera=itera)
        ####### Day Model 2######
        set_random_seed(seed)
        pprint('Min model fine tune...')
        pred_path = output_path+'/pred_%s_%d.pkl' %(model_name,itera)
        day_model_2 = getattr(models_module, model_name).Day_Model_2()
        _run = SummaryWriter(comment='_day_model2_min_%d' %
                             itera) if run_on else None
        rmse_min, mae_min = Day_model_2(
            _run,
            output_path,
            day_model_2,
            day_train_set,
            day_valid_set,
            day_test_set,
            min_train_day_reps,
            min_valid_day_reps,
            min_test_day_reps,
            pred_path)

        pprint(f'Iter: {itera}')
        pprint(
            f'pre_rmse_min: {pre_rmse_min}, rmse_min:{rmse_min}')
        pprint('###################')

if __name__ == '__main__':

    ex.run_commandline()