import sys
import g
#import g_torch_utils
np = g.np
import ExperimentManager as EM
import ast
import analyze

'''
example:
    
    python thisfile.py results_file ARS_config_file PARAM_NAME1=VALUE1 PARAM_NAME2=VALUE2


'''


if __name__=='__main__':
    default_hparams = {'q':0.01, 'tol':0.01, 'initDirection':-1, 'verbose':3, 'method':3, 'maxIters':1, 
                       'pin':'/extra/gurban0/DATA/PROTEOMICS/PSM_John_Halloran/SARS2/ihling2020_humanAndSars2.pin',
                       'output_dir':'/extra/gurban0/DATA/PROTEOMICS/PSM_John_Halloran/ARS_model_output/SARS2/',
                       'seed':1,
                        'dnn_optimizer': 'adam', 'batchsize': 5000, 'dnn_num_epochs': 2000, 
                        'dnn_lr': 0.001, 'l2_reg_const':0,
                        'dnn_num_layers':4, 'dnn_layer_size':300, 'dnn_dropout_rate':0.2, 'save_predictions':0,
                        'dnn_lr_decay':0.02, 'dnn_gpu_id':0, 'snapshot_ensemble_count':10,
                        'dnn_label_smoothing_0':1, 'dnn_label_smoothing_1':1, 'dnn_train_qtol':0.0025,
                        'false_positive_loss_factor':1.5}

    this_HP = default_hparams.copy()
#    this_HP.update(hparams)
    
    if len(sys.argv) <= 2:
        print('len(sys.argv) <= 2: running inactive experiments')

        results_file='results_file__new.txt'
        if not g.isfile(results_file):
            EM.create_experiment(this_HP, results_file=results_file)

        EM.main_loop(analyze.mainIter, None, default_hparams, results_file_naming_convention='results', results_file_search_dir='results/')
    else:
        assert len(sys.argv) >= 3, 'Format is:   python <thisfile.py> results_file ARS_config_file [PARAM_NAME1=VALUE1] [PARAM_NAME2=VALUE2] [...]'
        
        RES_FILE_NAME = sys.argv[1]
        ARS_CONFIG_FILE = sys.argv[2]
        print('RES_FILE_NAME:', RES_FILE_NAME)
        print('ARS_CONFIG_FILE:', ARS_CONFIG_FILE)
        if len(sys.argv) > 3:
            for X in sys.argv[3:]:
                k, v = X.split('=')
                try:
                    this_HP[k] = ast.literal_eval(v)
                except:
                    this_HP[k] = v
        
        EM.main_loop_ARS(analyze.mainIter, ARS_config_file=ARS_CONFIG_FILE, 
                         output_results_file=RES_FILE_NAME, valid_hyperparameters=None, 
                         default_hparams=this_HP, higher_is_better=True, num_iterations=250)
