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

from deepMs import calcQAndNumIdentified

_DEFAULT_HYPERPARAMS = {'dnn_optimizer': 'adam', 'batchsize': 2500, 'dnn_num_epochs': 500, 
                        'dnn_lr': 0.001, 'l2_reg_const':0,
                        'dnn_num_layers':3, 'dnn_layer_size':264, 'dnn_dropout_rate':0.2,
                        'dnn_lr_decay':0.2, 'dnn_gpu_id':0, 'snapshot_ensemble_count':10,
                        'dnn_label_smoothing_0':1, 'dnn_label_smoothing_1':1, 'dnn_train_qtol':0.002}


def q_val_AUC(qTol=0.003):
    def fn_auc(scores, labels):
        if labels.ndim==2:
            labels = np.argmax(labels, axis=1)
        if scores.ndim==2:
            scores = np.argmax(scores, axis=1)
        qs, ps = calcQAndNumIdentified(scores, labels)
        numIdentifiedAtQ = 0
        quac = []
        den = float(len(scores))
    #    ind0 = -1    
        for ind, (q, p) in enumerate(zip(qs, ps)):
            if q > qTol:
                break
            numIdentifiedAtQ = float(p)
            quac.append(numIdentifiedAtQ / den)
    #        if q < qCurveCheck:
    #            ind0 = ind
        # print "Accuracy = %f%%" % (numIdentifiedAtQ / float(len(qs)) * 100)
        # set AUC weights to uniform 
        auc = np.trapz(quac)#/len(quac)#/quac[-1]
    #    if qTol > qCurveCheck:
    #        auc = 0.3 * auc + 0.7 * np.trapz(quac[:ind0])#/ind0#/quac[ind0-1]
        return auc
    return fn_auc



class label_smoothing_loss(nn.Module):
    def __init__(self, device, class_confidence_values=[1, 1], class_weights = [1, 1]):
        """
        KL-divergence with label smoothing.
        
        class_confidence_values:
            
            array (length must be num_classes) in range 0...1;
            values of 1 == no smoothing; 
            0 == inverse training, where 'correct' class will have zero mass (don't do this...); 
            0.5 == only half weight to correct class, rest over others.
            
        class_weights:
            
            applied on a per-sample basis to weight specific classes up or down.
            
        """
        class_confidence_values = np.asarray(class_confidence_values, 'float32')
        assert np.ndim(class_confidence_values) == 1
        assert len(class_confidence_values) > 1
        assert len(class_weights) == len(class_confidence_values)
        assert np.all(0 <= class_confidence_values) and np.all( class_confidence_values <= 1)
        
        super(label_smoothing_loss, self).__init__()
        self.device = device
        self.amount = class_confidence_values
        self.num_classes = len(class_confidence_values)
        
        self.soft_distribs = np.zeros((self.num_classes, self.num_classes), 'float32')
        self.class_weights = np.asarray(class_weights, 'float32')
        for i in range(self.num_classes):
            other_amount = (1 - self.amount[i]) / (self.num_classes - 1)
            self.soft_distribs[i, :] = other_amount
            self.soft_distribs[i, i] = self.amount[i]
        
    def forward(self, output, labels):
        """
        output (float) shape: (batch_size, num_classes)
        labels (long) shape: (batch_size,)
        """
        labels = torch_utils.torch_tensor_to_np(labels)
        tmp2 = torch_utils.numpy_to_pytorch_tensor(self.class_weights[labels], device=self.device)
        soft_labels = np.zeros((len(labels), self.num_classes), 'float32')
        soft_labels[np.arange(len(labels)), :] = self.soft_distribs[labels]        
        tmp = F.kl_div(F.log_softmax(output, 1), torch_utils.numpy_to_pytorch_tensor(soft_labels, device=self.device), reduction='none').mean(1) #'batchmean')
#        if 0:
#            #un-weighted loss'
#            return (tmp).sum() / len(labels)
#        else:
        return (tmp2 * tmp).sum() / len(labels)

#TEST_X = label_smoothing_loss([0.95, 0.85])
#TEST_X.forward(0, [0,1,1,0,0,0,1]) 


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
        for lay in self._layers_MLP:
            x = torch.relu(lay(x))
            x = self.dropout(x)
        x = self._layer_output(x)
        if self._use_sigmoid_outputs:
            return torch.sigmoid(x)
        else:
            return x # if self.training==True else F.softmax(x, dim=1)

    
#def preprocess_data(X):
#    '''
#    zero mean, unit standard deviation
#    '''
#    return (X - X.mean(0)[None, :])/(X.std(0)[None, :] + 1e-15)



class ModelWrapper_like_sklearn(object):
    def __init__(self, model, device, batchsize=500):
        self._model = model
        self._device = device
        self._batchsize = batchsize
    
    def decision_function(self, X):
        return torch_utils.run_model_on_data(X, self._model, self._device, self._batchsize)[:,1]



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
    
    if 0:
        data_name = hparams['pin'].split('/')[-2]
        import g
        if not g.isfile('valid_data_{}_iter_{}.h5'.format(data_name, kFold)):
            g.save_list_h5('train_data_{}_iter_{}.h5'.format(data_name, kFold), train_data, ['data', 'labels'], 1, 1)
            g.save_list_h5('valid_data_{}_iter_{}.h5'.format(data_name, kFold), valid_data, ['data', 'labels'], 1, 1)
    
    if hparams['dnn_optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hparams['dnn_lr'])
    elif hparams['dnn_optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=hparams['dnn_lr'], momentum=0.9, weight_decay=hparams['l2_reg_const'])
    else:
        raise ValueError('optimizer {} not supported'.format(hparams['dnn_optimizer']))

#    val_metric = q_val_AUC(qTol=hparams['dnn_train_qtol'])
    val_metric = mini_utils.AUC_up_to_tol_singleQ(qTol=hparams['dnn_train_qtol'])
    
    # deal with trainin class imbalance
    n = len(train_data[1])
    n1 = np.sum(train_data[1])
    n2 = n - n1
    class_1_weight = n / 2. / n1
    class_2_weight = n / 2. / n2
    class_weights = (class_1_weight, class_2_weight)
    print('class_weights', class_weights)
    #loss_fn = nn.CrossEntropyLoss(),
    model, (train_acc, val_acc, test_acc), (train_loss_per_epoch, validation_loss_per_epoch) = torch_utils.train_model(
            model, DEVICE, loss_fn = label_smoothing_loss(DEVICE, [hparams['dnn_label_smoothing_0'], hparams['dnn_label_smoothing_1']], class_weights=class_weights), 
            optimizer=optimizer, train_data=train_data, 
            valid_data=valid_data, test_data=valid_data, 
            batchsize=hparams['batchsize'], num_epochs=hparams['dnn_num_epochs'], train=True, initial_lr=hparams['dnn_lr'], 
            total_lr_decay=hparams['dnn_lr_decay'], verbose=1, use_early_stopping=True, 
            validation_metric=val_metric, validation_check_interval=20,
            snapshot_ensemble_count=hparams['snapshot_ensemble_count'])
    
    #torch.save(the_model.state_dict(), 'output/' + 'MLP_model_params.h5')
    # grab predictions for class 1
    test_pred = torch_utils.run_model_on_data(valid_data[0], model, DEVICE, 5000)[:, 1]
        
    tp, _, _ = deepMs.calcQ(test_pred, validation_Labels, thresh, skipDecoysPlusOne=True)
    print("DNN CV finished for fold %d: %d targets identified" % (kFold, len(tp)))
    return test_pred, len(tp), ModelWrapper_like_sklearn(model, DEVICE)
