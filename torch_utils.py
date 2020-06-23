"""
Written by Gregor Urban <gur9000@outlook.com>

Copyright (C) 2020 Gregor Urban
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0
"""

import numpy as np
import time
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim


def torch_tensor_to_np(tensor):
    return tensor.data.cpu().numpy()


def softmax(x):
    """Compute softmax values for each sets of scores in x. (numpy)
    """
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, None] + 1e-16)


def numpy_to_pytorch_tensor(np_array, dtype=None, device='cpu', requires_grad=False):
    '''
    dtype can be:
        None, torch.float32, torch.long (or any other torch dtype)
    '''
    if dtype is None:
        dtype = torch.float32 if 'float' in str(np_array.dtype) else torch.long
    return torch.tensor(np_array, dtype=dtype, device=device, requires_grad=requires_grad)



def to_categorical(labels, num_classes = None):
    '''
    flat list (integers, values 0 and up) -> 2D tensor (num_samples, num_classes)
    '''
    labels = np.asarray(labels, 'int32')
    if not num_classes:
        num_classes = np.max(labels) + 1
    ret = np.zeros( (len(labels), num_classes), 'int32')
    for i, x in enumerate(labels):
        ret[i, x]=1
    return ret



def permute_data_2(data_list, seed=None, return_permutation=False, permutation = None):
    """
    Permutes a group of arrays (or lists) with the same permutation.
    To permute a single array, pass it as a list with one element (i.e. data_list = [my_array] )


    Returns:

        list of permuted data arrays,  [permutation if return_permutation==True]
    """
    if seed is not None:
        np_random_state = np.random.get_state()
        np.random.seed(int(seed))
    s = len(data_list[0])
    if permutation is None:
        per = np.random.permutation(np.arange(s))
    else:
        per = permutation
    ret = []
    for x in data_list:
        if isinstance(x, list) or isinstance(x, tuple):
            cpy = [x[i] for i in per]
        else:
            cpy = x[per]    #creates a copy! (fancy indexing)
        ret.append(cpy)
    if len(data_list)==1:
        ret = ret[0]
    if seed is not None:
        np.random.set_state(np_random_state)
    if not return_permutation:
        return ret
    else:
        return ret, per



def accuracy(predictions, labels):
    if labels.ndim==2:
        labels = np.argmax(labels, axis=1)
    if predictions.ndim==2:
        predictions = np.argmax(predictions, axis=1)
    return 100. * np.mean(predictions.astype('int32') == labels.astype('int32'))



def make_ensemble__greedy(list_of_predictions, labels, max_N_models_in_ensemble = 20, metric = accuracy):
    '''
    repeats: add one model to current ensemble so that ensemble accuracy/metric increases with each addition.
    Stops iteration if no improvement possible or if <max_N_models_in_ensemble> models were added (repetitions are counted).

    input:
    -------------
        list_of_predictions: probability/softmax predictions for validation set for all models.

        metric: function; higher values are better

    returns:
    -------------
        ensemble_predictions (combined/averaged predictions), ensemble_selection_indices
    '''
    ensemble_selection_indices = []
    ensemble_predictions = np.zeros(labels.shape, 'float32') #sum of model predictions
    for i in range(max_N_models_in_ensemble):
        j_best = 0
        score_best = -1e30
        for j,p in enumerate(list_of_predictions):
            combined_p = (ensemble_predictions + p) / (1.+len(ensemble_selection_indices))
            combined_score = metric(combined_p, labels)
            #print('make_ensemble__greedy:: iter',i,'score =',combined_score)
            if score_best < combined_score:
                score_best = combined_score
                j_best     = j
        if score_best <= -1e30:
            break #done, cannot improve
        ensemble_selection_indices.append(j_best)
        ensemble_predictions += list_of_predictions[j_best]
    return ensemble_predictions/len(ensemble_selection_indices), ensemble_selection_indices



def convert_data_dicts_to_torch(all_data, device='cpu'):
    '''
    list of dicts of np-arrays.
    '''
    for x in all_data:
        for k, v in x.items():
            x[k] = numpy_to_pytorch_tensor(v, dtype=torch.float32 if 'float' in str(v.dtype) else torch.long, device=device)



def predict(data, model):
    '''
    Returns a tensor containing the DNN's predictions for the given list of batches <data>.

    data must be a list of batches.
    '''
    model.eval()
    assert isinstance(data, list)
    pred = []
    for batch in data:
        if len(batch)==2:
            batch = batch[0]
        pred.append(model(batch))
    model.train()
    return np.concatenate(pred)



def get_model_params(model):
    weight_values = {}
    for k,v in model.state_dict().items():
        weight_values[k] = torch_tensor_to_np(v).copy()
    return weight_values



def set_model_params(model, state_dict):
    for k in state_dict:
        if type(state_dict[k]) is np.ndarray:
            state_dict[k] = numpy_to_pytorch_tensor(state_dict[k])
    model.load_state_dict(state_dict)
    return



def register_params_in_model(model, param_list, prefix=''):
    '''
    model must inherit from 'nn.Module'.

    Beware: will not distinguish between weights and biases.
    '''
    param_names = ['weight_{}_{}'.format(prefix, i) for i in range(len(param_list))]

    for name, param in zip(param_names, param_list):
        setattr(model, name, param) # model.param_name = nn.Parameter()
    model._all_weights = param_names



def update_lr(optimizer, initial_lr, relative_progress, total_lr_decay):
    """
    exponential decay

    initial_lr: any float (most reasonable values are in the range of 1e-5 to 1)
    total_lr_decay: value in (0, 1] -- this is the relative final LR at the end of training
    relative_progress: value in [0, 1] -- current position in training, where 0 == beginning, 1==end of training and a linear interpolation in-between
    """
    assert total_lr_decay > 0 and total_lr_decay <= 1
    lr = initial_lr * total_lr_decay**(relative_progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def run_model_on_data(data, model, device, batchsize=50):
    '''
    Returns the model's predictions as nd-array.
    
    Assumes that data is a numpy tensor.

    '''
    if hasattr(model, 'run_model_on_data'):
        return model.run_model_on_data(data, batchsize)
    model.eval()
    #model.training = False
    rval = None
    N = len(data)
    if batchsize > N:
        batchsize = N
    with torch.no_grad():
        for offset in range(0, N, batchsize):
            p = torch_tensor_to_np(model(numpy_to_pytorch_tensor(data[offset : offset + batchsize], device=device))) #(variable, num_outputs)
            if rval is None:
                rval = np.zeros((N,)+tuple(p.shape[1:]), 'float32')
            rval[offset:offset+batchsize, ...] = p
    model.train()
    return rval



def train_model(model, device, loss_fn, optimizer, train_data, valid_data, test_data,
                validation_metric = accuracy, validation_check_interval=1, #apply_softmax_to_predictions = True,
                batchsize = 100, num_epochs = 100, train = True,
                 initial_lr=3e-3, total_lr_decay=0.2, verbose = 1,
                 use_early_stopping = True,
                 snapshot_ensemble_count = 0):
    """
    Main training loop for the DNN.

    Input:
    ---------
    
    loss_fn(model_pred, labels):
        
        must return a torch scalar (the loss); e.g. loss_fn = nn.CrossEntropyLoss() # with class weights: nn.CrossEntropyLoss(weight = numpy_to_pytorch_tensor([0.5, 2, 3], device='cpu'))
    
    optimizer:
        
        function; e.g. optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    train_data, valid_data, test_data:

        each is a 2-tuple: (data, labels)

    validation_metric:
        
        either a function fn(predictions, labels) or a list of functions. If a list is provided then the output of the first function is returned as output, but the value of all is shown in the log.
        
    validation_check_interval:
        
        compute validation score only every XX epochs (to speed up training).
    total_lr_decay:

        value in (0, 1] -- this is the inverse total LR reduction factor over the course of training

    verbose:

        value in [0,1,2] -- 0 print minimal information (when training ends), 1 shows training loss, 2 shows training and validation loss after each epoch

    use_early_stopping:

        return model with weights from best epoch as judged by validation set score. Ignored if using snapshot ensembles.

    snapshot_ensemble_count [int]:
        
        If value > 0 then changes learning rate schedule to sawblade pattern, resetting to the initial value on each reset and decreasing down to a factor of <total_lr_decay>.
        Performs <snapshot_ensemble_count> such resets and stores model weight snapshots at lr resets. 
        Returns a model with weights averaged from the final <snapshot_ensemble_count> snapshots.

    Returns:
    -----------

        model, (train_acc, val_acc, test_acc), (train_loss_per_epoch, validation_loss_per_epoch)
    """
    print('train_model(): train_data: {}, valid_data: {}, test_data: {}'.format(*list(len(x[0]) for x in [train_data, valid_data, test_data])))
    
    if train:
        train_loss_per_epoch, validation_loss_per_epoch = [], []
        if verbose>0:
            print('starting training...')
        assert isinstance(snapshot_ensemble_count, (int, bool))
        best_valid_acc = 0
        if snapshot_ensemble_count > 0:
            use_early_stopping = 0
            lr_reset_every_n_epochs = int(np.ceil(num_epochs / snapshot_ensemble_count))
            model_params_snapshots = [] # will be a list of tuples, where each tuple is: (val_score, model_params)
        else:
            model_params_at_best_valid = []
        
        # optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=None)
        
        times=[]
        for epoch in range(num_epochs):
            if snapshot_ensemble_count > 0:
                update_lr(optimizer, initial_lr, (epoch % lr_reset_every_n_epochs) * 1. / lr_reset_every_n_epochs, total_lr_decay)
            else:
                update_lr(optimizer, initial_lr, epoch*1./num_epochs, total_lr_decay)
            train_data = permute_data_2(train_data)
            losses=[]
            t0 = time.time()
            for i in range(0, len(train_data[0]), batchsize):
                optimizer.zero_grad()
                this_batch = [numpy_to_pytorch_tensor(x[i:i+batchsize], device=device) for x in train_data]
                outputs = model(this_batch[0])
                loss = loss_fn(outputs, this_batch[1])
                loss.backward()
                optimizer.step()
                #loss = model.train_on_batch(x=train_data[i], y=train_data[i]['labels'], check_batch_dim=False)
                losses.append(torch_tensor_to_np(loss))
            times.append(time.time()-t0)
            if epoch % validation_check_interval == 0:
                val_pred = run_model_on_data(valid_data[0], model, device, batchsize = 2 * batchsize)
                val_acc = validation_metric(val_pred, valid_data[1])
                if np.isnan(val_acc):
                    print('ERROR: train_model():: validation_metric has returned NaN - aborting training!')
                    return (-1, -1, -1), (train_loss_per_epoch, validation_loss_per_epoch)
                if val_acc > best_valid_acc:
                    best_valid_acc = val_acc
                    if use_early_stopping:
                        model_params_at_best_valid = get_model_params(model) #kept in RAM (not saved to disk as that is slower)
                if verbose > 0:
                    print('Epoch {}/{} completed with average loss {:6.4f}; validation {} = {:6.4f}'.format(epoch+1, num_epochs, np.mean(losses), validation_metric.__name__, val_acc))
            else:
                val_acc = -1
                if verbose > 0:
                    print('Epoch {}/{} completed with average loss {:6.4f}'.format(epoch+1, num_epochs, np.mean(losses)))
                
            if snapshot_ensemble_count > 0 and (epoch % lr_reset_every_n_epochs) == lr_reset_every_n_epochs - 1:
                model_params_snapshots.append(get_model_params(model))
#            if verbose > 0:
#                print('Epoch {}/{} completed with average loss {:6.4f}; validation {} = {:6.4f}'.format(epoch+1, num_epochs, np.mean(losses), validation_metric.__name__, val_acc))
            train_loss_per_epoch.append(np.mean(losses))
            validation_loss_per_epoch.append(val_acc)
        # exclude times[0] as it includes compilation time!
        times.pop(0)
        print('Training @ {:5.3f} epochs/h, {:5.3f} samples/s)'.format(3600./np.mean(times), len(train_data[0])/np.mean(times)))

    if use_early_stopping:
        set_model_params(model, model_params_at_best_valid)
    if snapshot_ensemble_count > 0:
        model = make_ensemble(model, model_params_snapshots, valid_data, validation_metric, device, batchsize = 2 * batchsize)
    train_acc = validation_metric(run_model_on_data(train_data[0], model, device, batchsize = 2 * batchsize), train_data[1])
    val_acc = validation_metric(run_model_on_data(valid_data[0], model, device, batchsize = 2 * batchsize), valid_data[1])
    test_acc = validation_metric(run_model_on_data(test_data[0], model, device, batchsize = 2 * batchsize), test_data[1])
    print('Training completed:')
    print('  Training set score = {:6.4f}'.format(train_acc))
    print('  Validation score = {:6.4f}'.format(val_acc))
    print('  Test score = {:6.4f}'.format(test_acc))
    return model, (train_acc, val_acc, test_acc), (train_loss_per_epoch, validation_loss_per_epoch)



class Ensemble_Wrapper(object):
    def __init__(self, model, weights_list, device):
        '''
        Does NOT support training the model; only predictions via __call__().
        
        Warning: will modify weights of the model - don't use it for anything else after making predictions.
        '''
        self._weights_list = weights_list
        self._model = model
        self._device = device
        
    def __call__(self, X):
        preds = None
        for w in self._weights_list:
            set_model_params(self._model, w)
            if preds is None:
                preds = self._model(X)
            else:
                preds += self._model(X)
        return preds / len(self._weights_list)
    
    def run_model_on_data(self, data, batchsize=50):
        '''
        Like __call__ but on large amounts of data.
        '''
        self.eval()
        preds = None
        for w in self._weights_list:
            set_model_params(self._model, w)
            P = run_model_on_data(data, self._model, self._device, batchsize=batchsize)
            if preds is None:
                preds = P
            else:
                preds += P
        self.train()
        return preds / len(self._weights_list)
        
    def eval(self):
        self._model.eval()
    
    def train(self):
        self._model.train()
    
    def state_dict(self):
        '''WARNING: not implemented correctly!'''
        print('WARNING (TODO): Ensemble_Wrapper.state_dict(): not implemented correctly; only returns one model, not entire ensemble!')
        return self._model.state_dict()
    


def make_ensemble(model, list_weights, validation_set, metric, device, batchsize=500):
    val_preds = []
    for x in list_weights:
        set_model_params(model, x)
        val_preds.append(run_model_on_data(validation_set[0], model, device, batchsize = batchsize))
    val_pred, ensemble_indices = make_ensemble__greedy(val_preds, labels=to_categorical(validation_set[1]) if validation_set[1].ndim==1 else validation_set[1], 
                                                          max_N_models_in_ensemble=len(list_weights), metric=metric)
    return Ensemble_Wrapper(model, [list_weights[i] for i in ensemble_indices], device)


