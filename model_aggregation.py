import numpy as np
import pandas as pd
from nds import ndomsort
import pygmo as pg
import copy

from fair_models import SimpleVoting
import experiment_utils

def nds_moo(models_df, n_selected = 10, with_acc = False):
    models_df['EO'] = -models_df['EO']
    models_df['DP'] = -models_df['DP']
    if 'Acc' in models_df.columns:
        models_df['Acc'] = -models_df['Acc']
    metrics = models_df.values.tolist()

    fronts = ndomsort.non_domin_sort(metrics)
    selected_indexes = []
    for front in fronts:
        hv = pg.hypervolume([list(s) for s in fronts[front]])
        
        if len(selected_indexes)==n_selected:
            break
        
        if len(fronts[front])+len(selected_indexes)<n_selected:
            selected_indexes+=[metrics.index(seq) for seq in fronts[front]]
        else:
            last_front = list(copy.copy(fronts[front]))
            
            nadir = np.max(metrics,axis=0)
            while len(last_front)>n_selected-len(selected_indexes):
                hv = pg.hypervolume([list(s) for s in last_front])
                try:
                    idx_excl = hv.least_contributor(nadir)
                    del last_front[idx_excl]
                except:
                    break
                
            selected_indexes += [metrics.index(seq) for seq in last_front]
            
    index_list = [models_df.index.tolist()[i] for i in selected_indexes]
    return index_list


def ensemble_from_filtered_models(filtered_models, fair_feature, X_test, y_test):
    # Generate ensemble
    ensemble_model = SimpleVoting(estimators=filtered_models, voting='soft')
    # Evaluate
    metrics = experiment_utils.evaluate_model_test(ensemble_model, fair_feature, X_test, y_test)
    return metrics

def simple_filter(models_df, n_acc, fair_metric, fair_filter):
    if fair_filter:
        n_fair = 10
        if fair_metric == 'CV':
            index_list = list(models_df.nlargest(n_acc,'Acc').nsmallest(n_fair,fair_metric).index)
        else:
            index_list = list(models_df.nlargest(n_acc,'Acc').nlargest(n_fair,fair_metric).index)
    else:
        index_list = list(models_df.nlargest(n_acc,'Acc').index)
    return index_list

def nds_filter(models_df, n_acc, with_acc):
    models_acc = models_df.nlargest(n_acc,'Acc')
    if not with_acc:
        models_acc = models_acc.drop(['Acc'], axis=1)
    index_list = nds_moo(models_acc, with_acc = with_acc)
    return index_list

def ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc, fair_metric = 'DP', fair_filter = True,
                    nds = False, with_acc = True):
    if nds:
        index_list = nds_filter(models_df, n_acc, with_acc)
    else:
        index_list = simple_filter(models_df, n_acc, fair_metric, fair_filter)
    
    filtered_models = [("Model "+str(i), models_sols[i]) for i in index_list]
    metrics = ensemble_from_filtered_models(filtered_models, fair_feature, X_test, y_test)
    
    if nds:
        if n_acc==150:
            metrics['Filter'] = 'NDS(wAcc)' if with_acc else 'NDS'
        else:
            metrics['Filter'] = str(n_acc)+'Acc+NDS(wAcc)' if with_acc else str(n_acc)+'Acc+NDS'
    else:
        if fair_filter:
            metrics['Filter'] = str(n_acc)+'Acc+'+fair_metric
        else:
            metrics['Filter'] = 'All models' if n_acc==150 else str(n_acc)+'Acc'

    return metrics

def compare_ensembles_fair_metrics(models_df, models_sols, fair_feature, X_test, y_test):
    ensembles_metrics = [
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 150, fair_filter = False),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 50, fair_filter = True, fair_metric='DP'),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 20, fair_filter = True, fair_metric='DP'),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 50, fair_filter = True, fair_metric='EO'),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 20, fair_filter = True, fair_metric='EO'),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 50, fair_filter = True, fair_metric='CV'),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 20, fair_filter = True, fair_metric='CV')
    ]
    results_test = pd.DataFrame(ensembles_metrics)
    results_test = results_test.set_index('Filter')
    return results_test.copy()

def compare_ensembles_nds(models_df, models_sols, fair_feature, X_test, y_test):
    ensembles_metrics = [
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 150, fair_filter = False),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 150, nds = True, with_acc=True),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 50, nds = True, with_acc=True),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 20, nds = True, with_acc=True),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 150, nds = True, with_acc=False),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 50, nds = True, with_acc=False),
        ensemble_filter(models_df, models_sols, fair_feature, X_test, y_test, n_acc = 20, nds = True, with_acc=False)
    ]
    results_test = pd.DataFrame(ensembles_metrics)
    results_test = results_test.set_index('Filter')
    return results_test.copy()