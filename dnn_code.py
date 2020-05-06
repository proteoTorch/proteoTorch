"""
Written by Gregor Urban <gur9000@outlook.com>

Copyright (C) 2020 Gregor Urban
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0
"""
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import torch_utils

import mini_utils
import deepMs



_DEFAULT_HYPERPARAMS = {'dnn_optimizer': 'adam', 'batchsize': 2500, 'dnn_num_epochs': 500, 
                        'dnn_lr': 0.001, 'l2_reg_const':0,
                        'dnn_num_layers':3, 'dnn_layer_size':264, 'dnn_dropout_rate':0.2,
                        'dnn_lr_decay':0.2, 'dnn_gpu_id':0, 'snapshot_ensemble_count':10}



class MLP_model(nn.Module):
    def __init__(self, num_input_channels=15, number_of_classes = 2, use_sigmoid_outputs=False, 
                 dnn_num_layers = 3, dnn_layer_size = 100, dnn_dropout_rate = 0.2, **ignored):
        """        
        use_sigmoid_outputs:
            
            beware that this can mess up the loss function.
        """
        dnn_layer_sizes = [dnn_layer_size] * dnn_num_layers
        
        super(MLP_model, self).__init__()
        self._use_sigmoid_outputs = use_sigmoid_outputs
        assert isinstance(dnn_layer_sizes, list)
        self._layers_MLP = []
        n_in = num_input_channels
        self.dropout = nn.Dropout(p=dnn_dropout_rate)
        for i, nhid in enumerate(dnn_layer_sizes):
            lay = nn.Linear(n_in, nhid)
            n_in = nhid
            self._layers_MLP.append(lay)
        self._layer_output = nn.Linear(n_in, number_of_classes)
        params = []
        for x in self._layers_MLP:
            params.extend(x.parameters())
        torch_utils.register_params_in_model(self, params, 'MLP_weights') 

    def __call__(self, x):
        # x = inputs['input_features']
        for lay in self._layers_MLP:
            x = torch.relu(lay(x))
            x = self.dropout(x)
        x = self._layer_output(x)
        if self._use_sigmoid_outputs:
            return torch.sigmoid(x)
        else:
            return x if self.training else F.softmax(x)

    

def process_data(X):
    '''
    shift to 0..1 range.
    '''
    mi,ma = X.min(0)[None, :], X.max(0)[None, :]
    return (X - mi)/(ma - mi + 1e-7)



class ModelWrapper_like_sklearn(object):
    def __init__(self, model, device, batchsize=500):
        self._model = model
        self._device = device
        self._batchsize = batchsize
    
    def decision_function(self, X):
        return mini_utils.softmax(torch_utils.run_model_on_data(X, self._model, self._device, self._batchsize))[:,1]



def convert_labels(binary_labels):
    """
    input: list with two unique values, e.g. -1 and 1
    output: int32 numpy array with values 0, 1.
    """
    labels = np.asarray(binary_labels).astype('int64')
    labels -= labels.min()
    labels = labels / labels.max()
    return labels.astype('int64')



def DNNSingleFold(thresh, kFold, train_features, train_labels, validation_Features, validation_Labels, hparams = {}):
    """ 
    Train & test MLP model on one CV split
    
    hparams:
        
        dictionary with keys as found in _DEFAULT_HYPERPARAMS, or a subset of those keys: all missing keys will be mapped to the default values.
    """
    tmp_hparams = _DEFAULT_HYPERPARAMS.copy()
    tmp_hparams.update(hparams)
    hparams = tmp_hparams.copy()
    DEVICE = torch.device("cuda:"+str(hparams['dnn_gpu_id']) if torch.cuda.is_available() else "cpu")
    print('DNNSingleFold: working on device', DEVICE)
    model = MLP_model(num_input_channels=len(train_features[0]), number_of_classes = 2, **hparams)
    model = model.to(DEVICE)
    train_data = (np.asarray(train_features).astype('float32'), convert_labels(train_labels))
    valid_data = (np.asarray(validation_Features).astype('float32'), convert_labels(validation_Labels))
    
    if hparams['dnn_optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hparams['dnn_lr'])
    elif hparams['dnn_optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=hparams['dnn_lr'], momentum=0.9, weight_decay=hparams['l2_reg_const'])
    else:
        raise ValueError('optimizer {} not supported'.format(hparams['dnn_optimizer']))
    
    (train_acc, val_acc, test_acc), (train_loss_per_epoch, validation_loss_per_epoch) = torch_utils.train_model(
            model, DEVICE, loss_fn = nn.CrossEntropyLoss(), optimizer=optimizer, train_data=train_data, 
            valid_data=valid_data, test_data=valid_data, 
            batchsize=hparams['batchsize'], num_epochs=hparams['dnn_num_epochs'], train=True, initial_lr=hparams['dnn_lr'], 
            total_lr_decay=hparams['dnn_lr_decay'], verbose=1, use_early_stopping=True, 
            validation_metric=mini_utils.AUC_up_to_tol, snapshot_ensemble_count=hparams['snapshot_ensemble_count'])
    
    #torch.save(the_model.state_dict(), 'output/' + 'MLP_model_params.h5')
    # grab predictions for class 1
    test_pred = torch_utils.run_model_on_data(valid_data[0], model, DEVICE, 5000)[:, 1]
        
    tp, _, _ = deepMs.calcQ(test_pred, validation_Labels, thresh, skipDecoysPlusOne=True)
    print("DNN CV finished for fold %d: %d targets identified" % (kFold, len(tp)))
    return test_pred, len(tp), ModelWrapper_like_sklearn(model, DEVICE)
