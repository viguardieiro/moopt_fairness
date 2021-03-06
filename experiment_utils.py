import pandas as pd
import math
import random
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklego.metrics import equal_opportunity_score
from sklego.metrics import p_percent_score
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.extmath import squared_norm
from sklearn.model_selection import KFold
import optuna, sklearn, sklearn.datasets

from sklego.linear_model import DemographicParityClassifier
from sklego.linear_model import EqualOpportunityClassifier
from sklearn.linear_model import LogisticRegression

from moopt.scalarization_interface import scalar_interface, single_interface, w_interface
from moopt import monise

from fair_models import coefficient_of_variation, MOOLogisticRegression, FindCLogisticRegression, FindCCLogisticRegression
from fair_models import calc_reweight
from fair_models import FairScalarization, EqualScalarization
from fair_models import SimpleVoting

import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import sys
sys.path.append("./MMFP/")
from MMPF.MinimaxParetoFair.MMPF_trainer import SKLearn_Weighted_LLR, APSTAR

def evaluate_model_test(model__, fair_feature, X_test, y_test):
    return {"Acc": accuracy_score(y_test, model__.predict(X_test)),
            "EO": equal_opportunity_score(sensitive_column=fair_feature)(model__, X_test, y_test),
            "DP": p_percent_score(sensitive_column=fair_feature)(model__,X_test),
            "CV": coefficient_of_variation(model__, X_test, y_test)}


def evaluate_logreg(fair_feature, X_train, y_train, X_test, y_test):
    # Train
    logreg_model = LogisticRegression().fit(X_train, y_train)

    # Evaluate
    logreg_metrics = evaluate_model_test(logreg_model, fair_feature, X_test, y_test)
    logreg_metrics['Approach'] = 'LogReg'

    return logreg_metrics

def evaluate_reweigh(fair_feature, X_train, y_train, X_test, y_test):
    # Train
    sample_weight = calc_reweight(X_train, y_train, fair_feature)
    reweigh_model = LogisticRegression().fit(X_train, y_train,sample_weight=sample_weight)

    # Evaluate
    reweigh_metrics = evaluate_model_test(reweigh_model, fair_feature, X_test, y_test)
    reweigh_metrics['Approach'] = 'Reweigh'

    return reweigh_metrics

def evaluate_dempar(fair_feature, X_train, y_train, X_test, y_test):
    # Train
    dempar_model = DemographicParityClassifier(sensitive_cols=fair_feature, covariance_threshold=0)
    dempar_model.fit(X_train, y_train)

    # Evaluate
    dempar_metrics = evaluate_model_test(dempar_model, fair_feature, X_test, y_test)
    dempar_metrics['Approach'] = 'DemPar'

    return dempar_metrics

def evaluate_eqop(fair_feature, X_train, y_train, X_test, y_test):
    # Train
    eqop_model = EqualOpportunityClassifier(sensitive_cols=fair_feature, positive_target=True, covariance_threshold=0)
    eqop_model.fit(X_train, y_train)

    # Evaluate
    eqop_metrics = evaluate_model_test(eqop_model, fair_feature, X_test, y_test)
    eqop_metrics['Approach'] = 'EqOp'

    return eqop_metrics

def evaluate_minimax(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    # Train
    a_train = X_train[fair_feature].copy().astype('int')
    a_val = X_val[fair_feature].copy().astype('int')

    a_train[a_train==0] = -1
    a_val[a_val==0] = -1

    minimax_model = SKLearn_Weighted_LLR(X_train.values, y_train.values,
                    a_train.values, X_val.values,
                    y_val.values, a_val.values)

    mua_ini = np.ones(a_val.max() + 1)
    mua_ini /= mua_ini.sum()
    results = APSTAR(minimax_model, mua_ini, niter=200, max_patience=200, Kini=1,
                            Kmin=20, alpha=0.5, verbose=False)
    mu_best_list = results['mu_best_list']

    mu_best = mu_best_list[-1]
    minimax_model.weighted_fit(X_train.values, y_train.values, a_train.values, mu_best)

    # Evaluate
    minimax_metrics = evaluate_model_test(minimax_model, fair_feature, X_test, y_test)
    minimax_metrics['Approach'] = 'Minimax'

    return minimax_metrics

def evaluate_mooerr(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test, metric='EO'):
    # Train
    ## Train 150 models
    moo_err = monise(weightedScalar=FairScalarization(X_train, y_train, fair_feature),
                    singleScalar=FairScalarization(X_train, y_train, fair_feature),
                    nodeTimeLimit=2, targetSize=150,
                    targetGap=0, nodeGap=0.05, norm=False)

    moo_err.optimize()

    ## Evaluate the models in val
    mooerr_values = []
    mooerr_sols = []

    for solution in moo_err.solutionsList:
        mooerr_sols.append(solution.x)
        mooerr_values.append(evaluate_model_test(solution.x, fair_feature, X_val, y_val))

    mooerr_df = pd.DataFrame(mooerr_values)

    mooerr_metrics = None

    for metric in ['DP', 'EO', 'CV']:
        ## Filter the 20 best models in Acc, then the 10 best in metric
        if metric == 'CV':
            index_list = list(mooerr_df.nlargest(20,'Acc').nsmallest(10,metric).index)
        else:
            index_list = list(mooerr_df.nlargest(20,'Acc').nlargest(10,metric).index)
        mooerr_filtered_models = [("Model "+str(i), mooerr_sols[i]) for i in index_list]

        ## Generate ensemble
        ensemble_moo_err = SimpleVoting(estimators=mooerr_filtered_models, voting='soft', minimax=False)

        # Evaluate
        mooerr_metric = evaluate_model_test(ensemble_moo_err, fair_feature, X_test, y_test)
        mooerr_metric['Approach'] = 'MOOErr - '+metric
        if mooerr_metrics is None:
            mooerr_metrics = [mooerr_metric]
        else:
            mooerr_metrics.extend([mooerr_metric])

    return mooerr_metrics

def evaluate_mooacep(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    # Train 150 models
    mooacep = monise(weightedScalar=EqualScalarization(X_train, y_train, fair_feature),
                singleScalar=EqualScalarization(X_train, y_train, fair_feature),
                nodeTimeLimit=2, targetSize=150,
                targetGap=0, nodeGap=0.01, norm=False)

    mooacep.optimize()

    # Evaluate the models
    mooacep_values_val = []
    mooacep_sols = []

    for solution in mooacep.solutionsList:
        mooacep_sols.append(solution.x)
        mooacep_values_val.append(evaluate_model_test(solution.x, fair_feature, X_val, y_val))

    mooacep_df = pd.DataFrame(mooacep_values_val)

    mooacep_metrics = None

    for metric in ['DP', 'EO', 'CV']:
        ## Filter the 20 best models in Acc, then the 10 best in metric
        if metric == 'CV':
            index_list = list(mooacep_df.nlargest(20,'Acc').nsmallest(10,metric).index)
        else:
            index_list = list(mooacep_df.nlargest(20,'Acc').nlargest(10,metric).index)
        mooacep_filtered_models = [("Model "+str(i), mooacep_sols[i]) for i in index_list]

        ## Generate ensemble
        ensemble_mooacep = SimpleVoting(estimators=mooacep_filtered_models, voting='soft', minimax=False)

        # Evaluate
        mooacep_metric = evaluate_model_test(ensemble_mooacep, fair_feature, X_test, y_test)
        mooacep_metric['Approach'] = 'MOOAcep - '+metric
        if mooacep_metrics is None:
            mooacep_metrics = [mooacep_metric]
        else:
            mooacep_metrics.extend([mooacep_metric])

    return mooacep_metrics

def evaluate_all_approaches(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    models_metrics = [evaluate_logreg(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_reweigh(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_dempar(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_eqop(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_minimax(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test)]
    models_metrics.extend(evaluate_mooerr(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test))
    models_metrics.extend(evaluate_mooacep(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test))

    return pd.DataFrame(models_metrics).set_index('Approach')

def kfold_methods(dataset, fair_feature, pred_feature, n_folds = 5):
    kfold = KFold(n_folds, True, 1)

    results_test = pd.DataFrame()

    for _ in range(n_folds):    
        result = next(kfold.split(dataset), None)

        train = dataset.iloc[result[0]].drop("Unnamed: 0", axis=1)
        test = dataset.iloc[result[1]].drop("Unnamed: 0", axis=1)

        categories_fair_class = []
        for index, row in train.iterrows():
            if row[pred_feature] == -1:
                categories_fair_class.append(row[fair_feature])
            else:
                categories_fair_class.append(row[fair_feature]+2)

        X = train.drop([pred_feature], axis=1)
        y = train[pred_feature]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=int(X.shape[0]*0.2), stratify=categories_fair_class)
        X_test = test.drop([pred_feature], axis=1)
        y_test = test[pred_feature]

        eval_result = evaluate_all_approaches(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test)

        results_test = pd.concat((results_test, eval_result))

    return results_test