import copy
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.utils import pprint, AverageMeter
from common.functions import get_loss_fn, get_metric_fn

from sacred import Ingredient
model_ingredient = Ingredient('cnn_rnn_v2')


@model_ingredient.config
def model_config():
    # architecture
    input_shape = [6, 60]
    rnn_type = 'LSTM'  # LSTM/GRU
    rnn_layer = 2
    hid_size = 64
    dropout = 0
    # optimization
    optim_method = 'Adam'
    optim_args = {'lr': 1e-3}
    loss_fn = 'mse'
    eval_metric = 'corr'
    verbose = 500
    max_steps = 50
    early_stopping_rounds = 5
    min_max_steps = 50
    min_early_stopping_rounds = 5
    min_loss_fn = 'mse_v2'
    min_ratio_teacher =  0.3
    min_optim_args = {'lr': 1e-3}
    output_path = "/home/amax/Documents/HM_CNN_RNN/out"

class Day_Model_1(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout=0,
                 optim_method='Adam',
                 optim_args={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr'):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_size = input_shape[0]
        self.input_day = input_shape[2]
        self.dropout = dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)
        self.net = nn.Sequential()
        self.net.add_module('fc_in',nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module('act',nn.Tanh())
        self.rnn = klass(input_size=self.hid_size,
                         hidden_size=self.hid_size,
                         num_layers=self.rnn_layer,
                         batch_first=True,
                         dropout=self.dropout)
        self.fc_final = nn.Linear(in_features=self.hid_size, out_features=1)
        
    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_size, self.input_day)
        inputs = inputs.permute(0, 2, 1) #[batch, input_size, seq_len] -> [batch, seq_len, input_size]
        fc_hid = self.net(inputs)
        out, _ = self.rnn(fc_hid)
        out_seq = self.fc_final(out[:, -1, :]) # [batch, input_day, hid_size] -> [batch, 1]
        
        return fc_hid, out_seq[..., 0]# fc_hid:[batch, input_day, hid_size]


    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            run=None,
            max_steps=50,
            early_stopping_rounds=10,
            verbose=100):

        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())
        for step in range(max_steps):
            
            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            train_hids = dict()
            for i, (idx, data, label) in enumerate(train_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                train_hid, pred = self(data)
                # pprint(f'train_hid: {train_hid.shape}')
                # pprint(f'pred: {pred.shape}')
                loss = self.loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                train_loss.update(loss_, len(data))
                train_eval.update(eval_)
                train_hids[idx] = train_hid.cpu().detach()
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            valid_hids = dict()
            for i, (idx, data, label) in enumerate(valid_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    valid_hid, pred = self(data)
                
                loss = self.loss_fn(pred, label)
                valid_loss_ = loss.item()
                valid_eval_ = self.metric_fn(pred, label).item()
                valid_loss.update(valid_loss_, len(data))
                valid_eval.update(valid_eval_)
                valid_hids[idx] = valid_hid.cpu().detach()
            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
        # restore
        self.load_state_dict(best_params)
        return train_hids, valid_hids # train_hid: [batch, input_day, hid_size]

    def predict(self, test_set):
        self.eval()
        preds = []
        test_hids = dict()
        for _, (idx, data, _) in enumerate(test_set):
            data = torch.tensor(data, dtype=torch.float)
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                test_hid, pred = self(data)
            test_hids[idx] = test_hid.cpu().detach()
            preds.append(pred.cpu().detach().numpy())
        return np.concatenate(preds), test_hids

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)

class InputConv(nn.Module):
    def __init__(self,
                 in_height,
                 in_width,
                 input_day,
                 out_chnls,
                 kernel,
                 stride):

        super().__init__()

        self.in_height = in_height
        self.in_width = in_width
        self.input_day = input_day
        self.out_chnls = out_chnls
        self.kernel = kernel
        self.stride = stride

        self._build_model()

    def _build_model(self):

        self.cnn_2d= nn.Conv2d(1, self.out_chnls[0], kernel_size=[self.in_height,1], stride=1, padding=0) #[-1,1,6,16] -> [-1,6,1,16]
        input_chnl = self.out_chnls[0]
        self.cnn = nn.Sequential()
        for i, out_chnl in enumerate(self.out_chnls[1:]):  # [6,6,6]
            self.cnn.add_module('conv_%d'%(i+1),
                    nn.Conv1d(input_chnl, out_chnl, kernel_size=self.kernel[i], stride=self.stride[i], padding=0)) #[6,16] -> [6,4] -> [6,1])
            self.cnn.add_module("tanh_%d"%(i+1), nn.Tanh())
            input_chnl = out_chnl
        # self.cnn.add_module('bn_%d'%(i+1), nn.BatchNorm1d(out_chnl))
        self.out_dim = out_chnl 

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_day, self.in_height, self.in_width)
        arr  = []
        for i in range(self.input_day):
            input = inputs[:,i,:,:] #[batch, input_day, input_size, input_length] -> [batch, input_size, input_length]
            input = input.unsqueeze(1) # [batch, input_size, input_length] -> [batch, 1, input_size, input_length]
            out = self.cnn_2d(input) # [batch, 1, input_size, input_length] -> [batch, out_chnls[0], 1, input_length]
            out = out.squeeze(2) # [batch, out_chnls[0], 1, input_length] -> [batch, out_chnls[0], input_length]
            out = self.cnn(out) #[batch, out_chnls[0], input_length] -> [batch, out_dim, 1]
            out = out.permute(0,2,1) #[batch, out_dim, 1] -> [batch, 1, out_dim]
            arr.append(out)
        day_reps = torch.cat(arr,dim=1) #arr: [batch, 1, out_dim] * input_day -> [batch, input_day, out_dim]
        # pprint(f'day_reps: {day_reps.shape}')
        return day_reps


class Mix_Model(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 mix_dropout=0,
                 optim_method='Adam',
                 mix_optim_args={'lr': 1e-3},
                 min_loss_fn='mse_v2',
                 eval_metric='corr',
                 out_chnls=[6,6,6],
                 kernel= [4,4],
                 stride=[4,4],
                 min_ratio_teacher = 0.3):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_size = input_shape[0]
        self.input_length = input_shape[1]
        self.input_day = input_shape[2]
        self.dropout = mix_dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.out_chnls = out_chnls
        self.kernel = kernel
        self.stride = stride
        self.min_ratio_teacher = min_ratio_teacher

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **mix_optim_args)
        self.loss_fn = get_loss_fn(min_loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)

        self.input_conv = InputConv(in_height = self.input_size,
                                    in_width = self.input_length,
                                    input_day = self.input_day,
                                    out_chnls = self.out_chnls,
                                    kernel = self.kernel,
                                    stride = self.stride)

        self.net = nn.Sequential()
        self.net.add_module('fc_in',nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module('act',nn.Tanh())

        self.rnn = klass(input_size=self.hid_size, 
                         hidden_size=self.hid_size,
                         num_layers=self.rnn_layer,
                         batch_first=True,
                         dropout=self.dropout)

        self.fc_final = nn.Linear(in_features=self.hid_size, out_features=1)
            
    def forward(self, inputs):
        input_cnn = self.input_conv(inputs) # [batch, input_day, out_dim]
        fc_hid = self.net(input_cnn)
        out, _ = self.rnn(fc_hid)
        out_seq = self.fc_final(out[:, -1, :]) # [batch, input_length, hid_size + free_hid] -> [batch, 1]
        return fc_hid, out_seq[..., 0]  # input_cnn: [batch, input_length, input_size]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            old_train_day_reps, # daily RNN hids
            old_valid_day_reps,
            run=None,
            min_max_steps=50,
            min_early_stopping_rounds=5,
            verbose=100,
            output_path = "/home/amax/Documents/HM_CNN_RNN/out",
            itera=0):

        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        # for step in range(max_steps):
        for step in range(min_max_steps): 
            pprint('Step:', step)
            # self.min_ratio_teacher = 1 / (step+1)
            # if self.min_ratio_teacher <= 0.1:
            #     self.min_ratio_teacher = 0 
            if stop_steps >= min_early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_loss_a = AverageMeter()
            train_loss_b = AverageMeter()
            train_eval = AverageMeter()
            train_eval_a = AverageMeter()
            train_eval_b = AverageMeter()
            train_day_reps = dict() # min -> day representation
            for i, (idx, data, label) in enumerate(train_set):

                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                old_train_day_rep = old_train_day_reps[idx] # train_hid:[batch, input_day, hid_size ]
                if torch.cuda.is_available():
                    data, old_train_day_rep, label  = data.cuda(), old_train_day_rep.cuda(), label.cuda()
                day_rep, pred = self(data)

                #[batch, input_day, hid_size]
                train_day_rep = day_rep[:,:,:self.hid_size]
                loss_a = self.loss_fn(train_day_rep, old_train_day_rep, dim=[0,1,2]) # learn from teacher
                loss_b = self.loss_fn(pred, label, dim=0) # learn from label 2
                # loss = self.min_ratio_teacher * loss_a + (1.0-self.min_ratio_teacher) * loss_b
                loss = self.min_ratio_teacher * loss_a + loss_b
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_day_reps[idx] = train_day_rep.cpu().detach()
                len_data = len(data)
                train_loss.update(loss.item(), len_data)
                if loss.item() > 1000:
                    pprint(idx)
                train_loss_a.update(loss_a.item(), len_data)
                train_loss_b.update(loss_b.item(), len_data)
                eval_a = self.metric_fn(train_day_rep, old_train_day_rep, dim=[0,1,2]).item()
                eval_b = self.metric_fn(pred, label,dim=0).item()
                # eval_ =  self.min_ratio_teacher * eval_a + (1.0-self.min_ratio_teacher) * eval_b
                eval_ = eval_b
                train_eval.update(eval_)
                train_eval_a.update(eval_a)
                train_eval_b.update(eval_b)
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_loss_a = AverageMeter()
            valid_loss_b = AverageMeter()
            valid_eval = AverageMeter()
            valid_eval_a = AverageMeter()
            valid_eval_b = AverageMeter()
            valid_day_reps = dict()
            for i, (idx, data, label) in enumerate(valid_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                old_valid_day_rep = old_valid_day_reps[idx]
                if torch.cuda.is_available():
                    data, old_valid_day_rep, label = data.cuda(), old_valid_day_rep.cuda(), label.cuda()
                with torch.no_grad():
                    day_rep, pred = self(data)
                valid_day_rep = day_rep[:,:,:self.hid_size]
                loss_a = self.loss_fn(valid_day_rep, old_valid_day_rep, dim=[0,1,2])
                loss_b = self.loss_fn(pred, label, dim=0)
                # loss = self.min_ratio_teacher * loss_a + (1.0-self.min_ratio_teacher) * loss_b
                loss = self.min_ratio_teacher * loss_a + loss_b
                valid_day_reps[idx] = valid_day_rep.cpu().detach()
                len_data = len(data)
                valid_loss.update(loss.item(), len_data)
                valid_loss_a.update(loss_a.item(), len_data)
                valid_loss_b.update(loss_b.item(), len_data)
                eval_a = self.metric_fn(valid_day_rep, old_valid_day_rep, dim=[0,1,2]).item()
                eval_b = self.metric_fn(pred, label,dim=0).item()
                # eval_ = self.min_ratio_teacher * eval_a + (1.0-self.min_ratio_teacher) * eval_b
                eval_ = eval_b
                valid_eval.update(eval_)
                valid_eval_a.update(eval_a)
                valid_eval_b.update(eval_b)
            if run is not None:
                run.add_scalar('Train/Loss_total', train_loss.avg, step)
                run.add_scalar('Train/Loss_from_teacher', train_loss_a.avg, step)
                run.add_scalar('Train/Loss_from_label', train_loss_b.avg, step)
                run.add_scalar('Train/Eval_total', train_eval.avg, step)
                run.add_scalar('Train/Eval_from_teacher', train_eval_a.avg, step)
                run.add_scalar('Train/Eval_from_label', train_eval_b.avg, step)

                run.add_scalar('Valid/Loss_total', valid_loss.avg, step)
                run.add_scalar('Valid/Loss_from_teacher', valid_loss_a.avg, step)
                run.add_scalar('Valid/Loss_from_label', valid_loss_b.avg, step)
                run.add_scalar('Valid/Eval_total', valid_eval.avg, step)
                run.add_scalar('Valid/Eval_from_teacher', valid_eval_a.avg, step)
                run.add_scalar('Valid/Eval_from_label', valid_eval_b.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_train_day_reps = train_day_reps
                best_valid_day_reps = valid_day_reps
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())

                self.save(output_path+'/best_mix_model_%d.bin'%itera)
        # restore
        self.load_state_dict(best_params)
        return best_train_day_reps, best_valid_day_reps

    def predict(self, test_set):
        self.eval()
        test_day_reps = dict()
        for _, (idx, data, _) in enumerate(test_set):
            data = torch.tensor(data, dtype=torch.float)
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                test_day_rep, _ = self(data)
            test_day_reps[idx] = test_day_rep.cpu().detach()
        return test_day_reps

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)


class Min_Model(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 min_dropout=0,
                 optim_method='Adam',
                 min_optim_args={'lr': 1e-3},
                 min_loss_fn='mse_v2',
                 eval_metric='corr',
                 min_ratio_teacher = 0.3):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_size = input_shape[0]
        self.input_length = input_shape[1]
        self.input_day = input_shape[2]
        self.min_dropout = min_dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.min_ratio_teacher = min_ratio_teacher

        self._build_model()

        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **min_optim_args)
        self.loss_fn = get_loss_fn(min_loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)

        self.rnn = klass(input_size=self.input_size,
                         hidden_size=self.hid_size ,
                         num_layers=self.rnn_layer,
                         batch_first=True,
                         dropout=self.min_dropout)

        self.daily_rnn = klass(input_size=self.hid_size,
                    hidden_size=self.hid_size ,
                    num_layers=self.rnn_layer,
                    batch_first=True,
                    dropout=self.min_dropout)

        self.fc_final = nn.Linear(in_features=self.hid_size, out_features=1)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_day, self.input_size, self.input_length)
        arr  = []
        for i in range(self.input_day):
            input = inputs[:,i,:,:] #[batch, input_day, input_size, seq_len] -> [batch, input_size, seq_len]
            input = input.permute(0,2,1)# [batch, input_size, seq_len] -> [batch, seq_len, input_size]

            out, _ = self.rnn(input) # [batch, seq_len, input_size] -> [batch, seq_len, hid_size ]
            out = out[:,-1,:].unsqueeze(1)#[batch, seq_len, hid_size] -> [batch, 1, hid_size ]
            arr.append(out)

        day_reps = torch.cat(arr,dim=1)#arr: [batch, 1, hid_size ] * input_day -> [batch, input_day, hid_size ]
        daily_out, _ = self.daily_rnn(day_reps)
        daily_out = self.fc_final(daily_out[:, -1, :])# [batch, input_day, hid_size ] -> [batch, 1]
        
        return day_reps, daily_out[...,0]
        #day_reps : [batch, input_day, hid_size]
        #daily_out : [batch]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            train_hids, # daily RNN hids
            valid_hids,
            run=None,
            min_max_steps=50,
            min_early_stopping_rounds=5,
            verbose=100,
            output_path = "/home/amax/Documents/HM_CNN_RNN/out",
            itera=0):

        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        # for step in range(max_steps):
        for step in range(min_max_steps): 
            # self.min_ratio_teacher = 1 / (step+1)
            # if self.min_ratio_teacher <= 0.1:
            #     self.min_ratio_teacher = 0 
            pprint('Step:', step)
            if stop_steps >= min_early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_loss_a = AverageMeter()
            train_loss_b = AverageMeter()
            train_eval = AverageMeter()
            train_eval_a = AverageMeter()
            train_eval_b = AverageMeter()
            train_day_reps = dict() # min -> day representation
            for i, (idx, data, label) in enumerate(train_set):
                self.optimizer.zero_grad()
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                train_hid = train_hids[idx] # train_hid:[batch, input_day, input_size]
                if torch.cuda.is_available():
                    data, train_hid, label  = data.cuda(), train_hid.cuda(), label.cuda()
                day_rep, pred = self(data)

                #[batch, input_day, hid_size ]
                train_day_rep = day_rep[:,:,:self.hid_size]
                # pprint(f'train_day_rep: {train_day_rep.shape}')
                # pprint(f'train_hid: {train_hid.shape}')
                loss_a = self.loss_fn(train_day_rep, train_hid, dim=[0,1,2]) # learn from teacher
                
                loss_b = self.loss_fn(pred, label, dim=0) # learn from label 2
                # loss = self.min_ratio_teacher * loss_a + (1.0-self.min_ratio_teacher) * loss_b
                loss = self.min_ratio_teacher * loss_a + loss_b
                
                loss.backward()
                self.optimizer.step()
                train_day_reps[idx] = day_rep.cpu().detach()
                len_data = len(data)
                train_loss.update(loss.item(), len_data)
                if loss.item() > 1000:
                    pprint(idx)
                train_loss_a.update(loss_a.item(), len_data)
                train_loss_b.update(loss_b.item(), len_data)
                eval_a = self.metric_fn(train_day_rep, train_hid, dim=[0,1,2]).item()
                eval_b = self.metric_fn(pred, label,dim=0).item()
                # eval_ = self.min_ratio_teacher * eval_a + (1.0-self.min_ratio_teacher) * eval_b
                eval_ = eval_b
                train_eval.update(eval_)
                train_eval_a.update(eval_a)
                train_eval_b.update(eval_b)
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_loss_a = AverageMeter()
            valid_loss_b = AverageMeter()
            valid_eval = AverageMeter()
            valid_eval_a = AverageMeter()
            valid_eval_b = AverageMeter()
            valid_day_reps = dict()
            for i, (idx, data, label) in enumerate(valid_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                valid_hid = valid_hids[idx]
                if torch.cuda.is_available():
                    data, valid_hid, label = data.cuda(), valid_hid.cuda(), label.cuda()
                with torch.no_grad():
                    day_rep, pred = self(data)
                valid_day_rep = day_rep[:,:,:self.hid_size]
                loss_a = self.loss_fn(valid_day_rep, valid_hid, dim=[0,1,2])
                loss_b = self.loss_fn(pred, label, dim=0)
                # loss = self.min_ratio_teacher * loss_a + (1.0-self.min_ratio_teacher) * loss_b
                loss = self.min_ratio_teacher * loss_a + loss_b
                valid_day_reps[idx] = day_rep.cpu().detach()
                len_data = len(data)
                valid_loss.update(loss.item(), len_data)
                valid_loss_a.update(loss_a.item(), len_data)
                valid_loss_b.update(loss_b.item(), len_data)
                eval_a = self.metric_fn(valid_day_rep, valid_hid,dim=[0,1,2]).item()
                eval_b = self.metric_fn(pred, label,dim=0).item()
                # eval_ = self.min_ratio_teacher * eval_a + (1.0-self.min_ratio_teacher) * eval_b
                eval_ = eval_b
                valid_eval.update(eval_)
                valid_eval_a.update(eval_a)
                valid_eval_b.update(eval_b)
            if run is not None:
                run.add_scalar('Train/Loss_total', train_loss.avg, step)
                run.add_scalar('Train/Loss_from_teacher', train_loss_a.avg, step)
                run.add_scalar('Train/Loss_from_label', train_loss_b.avg, step)
                run.add_scalar('Train/Eval_total', train_eval.avg, step)
                run.add_scalar('Train/Eval_from_teacher', train_eval_a.avg, step)
                run.add_scalar('Train/Eval_from_label', train_eval_b.avg, step)

                run.add_scalar('Valid/Loss_total', valid_loss.avg, step)
                run.add_scalar('Valid/Loss_from_teacher', valid_loss_a.avg, step)
                run.add_scalar('Valid/Loss_from_label', valid_loss_b.avg, step)
                run.add_scalar('Valid/Eval_total', valid_eval.avg, step)
                run.add_scalar('Valid/Eval_from_teacher', valid_eval_a.avg, step)
                run.add_scalar('Valid/Eval_from_label', valid_eval_b.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_train_day_reps = copy.deepcopy(train_day_reps)
                best_valid_day_reps = copy.deepcopy(valid_day_reps)
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())

                self.save(output_path+'/best_min_model_%d.bin'%itera)
        # restore
        self.load_state_dict(best_params)
        return best_train_day_reps, best_valid_day_reps

    def predict(self, test_set):
        self.eval()
        test_day_reps = dict()
        for _, (idx, data, _) in enumerate(test_set):
            data = torch.tensor(data, dtype=torch.float)
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                day_rep, _ = self(data)
            test_day_reps[idx] = day_rep.cpu().detach()
        return test_day_reps

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)

class Day_Model_2(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout_2=0,
                 optim_method='Adam',
                 optim_args_2={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr',
                 min_ratio_teacher=0.3):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_shape = input_shape
        self.input_size = input_shape[0] 
        self.input_day = input_shape[2]
        self.dropout = dropout_2
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args_2)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)

        self.net = nn.Sequential()
        self.net.add_module('fc_in',nn.Linear(in_features= self.hid_size, out_features = self.hid_size))
        self.net.add_module('act',nn.Tanh())

        self.rnn = klass(input_size=self.hid_size + self.input_size,
                hidden_size=self.hid_size + self.input_size,
                num_layers=self.rnn_layer,
                batch_first=True,
                dropout=self.dropout)
        self.fc_final = nn.Linear(in_features=self.hid_size + self.input_size, out_features=1)

    def forward(self, inputs, data):
        fc_hid = self.net(inputs)
        out, _ = self.rnn(torch.cat([fc_hid, data],dim=2))
        out_seq = self.fc_final(out[:, -1, :]) # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return fc_hid, out_seq[..., 0]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            train_day_reps,
            valid_day_reps,
            run=None,
            max_steps=100,
            early_stopping_rounds=10,
            verbose=100):
        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        for step in range(max_steps):
            
            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            tune_train_reps = dict()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(train_set):
                train_day_rep = train_day_reps[idx]
                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_size, self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    train_day_rep, label, data = train_day_rep.cuda(), label.cuda(), data.cuda()
                tune_train_rep, pred = self(train_day_rep, data)
                loss = self.loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_train_reps[idx] = tune_train_rep.cpu().detach()
                train_loss.update(loss_, len(label))
                train_eval.update(eval_)
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            tune_valid_reps = dict()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(valid_set):
                valid_day_rep = valid_day_reps[idx]
                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_shape[0], self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    valid_day_rep, label, data = valid_day_rep.cuda(), label.cuda(), data.cuda()
                with torch.no_grad():
                    tune_valid_rep, pred = self(valid_day_rep,data)
                loss = self.loss_fn(pred, label)
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_valid_reps[idx] = tune_valid_rep.cpu().detach()
                valid_loss.update(loss_, len(label))
                valid_eval.update(eval_)

            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)

            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
                best_tune_train_reps = tune_train_reps
                best_tune_valid_reps = tune_valid_reps

        # restore
        self.load_state_dict(best_params)
        return best_tune_train_reps, best_tune_valid_reps

    def predict(self, test_set, test_day_reps):
        self.eval()
        preds = []
        tune_test_reps = dict()
        for _, (idx, data, _) in enumerate(test_set):
            test_day_rep = test_day_reps[idx]
            
            data = torch.tensor(data, dtype=torch.float)
            data = data.view(-1, self.input_size, self.input_day)
            data = data.permute(0, 2, 1)
            if torch.cuda.is_available():
                test_day_rep, data = test_day_rep.cuda(), data.cuda()
            with torch.no_grad():
                tune_test_rep, pred = self(test_day_rep, data)
                tune_test_reps[idx] = tune_test_rep.cpu().detach()
                preds.append(pred.cpu().detach().numpy())
        return np.concatenate(preds), tune_test_reps

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)

class Day_Model_3(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout_3=0,
                 optim_method='Adam',
                 optim_args_3={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr',
                 min_ratio_teacher=0.3):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_shape = input_shape
        self.input_size = input_shape[0] 
        self.input_day = input_shape[2]
        self.dropout = dropout_3
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args_3)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)

        # self.net = nn.Sequential()
        # self.net.add_module('fc_in',nn.Linear(in_features=self.hid_size * 2, out_features = self.hid_size * 2))
        # self.net.add_module('act',nn.Tanh())

        self.rnn = klass(input_size=self.hid_size * 2 + self.input_size,
                    hidden_size=self.hid_size * 2 + self.input_size,
                    num_layers=self.rnn_layer,
                    batch_first=True,
                    dropout=self.dropout)

        self.fc_final = nn.Linear(in_features=self.hid_size * 2 + self.input_size, out_features=1)
            
    def forward(self, rnn_inputs, cnn_inputs, raw_inputs):
        fc_hid = rnn_inputs
        # fc_hid = self.net(torch.cat([rnn_inputs, cnn_inputs],dim=2))
        out, _ = self.rnn(torch.cat([rnn_inputs, cnn_inputs, raw_inputs], dim=2))
        out_seq = self.fc_final(out[:, -1, :]) # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return fc_hid, out_seq[..., 0]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            train_day_reps_rnn,
            train_day_reps_cnn,
            valid_day_reps_rnn,
            valid_day_reps_cnn,
            run=None,
            max_steps=100,
            early_stopping_rounds=10,
            verbose=100):
        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        for step in range(max_steps):
            
            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            tune_train_reps = dict()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(train_set):
                train_day_rep_rnn = train_day_reps_rnn[idx]
                train_day_rep_cnn = train_day_reps_cnn[idx]
                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_size, self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    train_day_rep_cnn, train_day_rep_rnn, label, data = train_day_rep_cnn.cuda(), \
                        train_day_rep_rnn.cuda(), label.cuda(), data.cuda()
                tune_train_rep, pred = self(train_day_rep_rnn, train_day_rep_cnn, data)
                loss = self.loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_train_reps[idx] = tune_train_rep.cpu().detach()
                train_loss.update(loss_, len(label))
                train_eval.update(eval_)
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            tune_valid_reps = dict()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(valid_set):
                valid_day_rep_rnn = valid_day_reps_rnn[idx]
                valid_day_rep_cnn = valid_day_reps_cnn[idx]

                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_size, self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    valid_day_rep_cnn, valid_day_rep_rnn, label, data = valid_day_rep_cnn.cuda(), \
                        valid_day_rep_rnn.cuda(), label.cuda(), data.cuda()
                tune_valid_rep, pred = self(valid_day_rep_rnn, valid_day_rep_cnn, data)
                loss = self.loss_fn(pred, label)
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_valid_reps[idx] = tune_valid_rep.cpu().detach()
                valid_loss.update(loss_, len(label))
                valid_eval.update(eval_)

            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)

            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
                best_tune_train_reps = tune_train_reps
                best_tune_valid_reps = tune_valid_reps

        # restore
        self.load_state_dict(best_params)
        return best_tune_train_reps, best_tune_valid_reps

    def predict(self, test_set, test_day_reps_rnn, test_day_reps_cnn):
        self.eval()
        preds = []
        for _, (idx, data, _) in enumerate(test_set):
            test_day_rep_rnn = test_day_reps_rnn[idx]
            test_day_rep_cnn = test_day_reps_cnn[idx]
            data = torch.tensor(data, dtype=torch.float)
            data = data.view(-1, self.input_size, self.input_day)
            data = data.permute(0, 2, 1)
            if torch.cuda.is_available():
                test_day_rep_rnn, test_day_rep_cnn, data = test_day_rep_rnn.cuda(), test_day_rep_cnn.cuda(), data.cuda()
            with torch.no_grad():
                _, pred = self(test_day_rep_rnn, test_day_rep_cnn, data)
                preds.append(pred.cpu().detach().numpy())
        return np.concatenate(preds)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)

