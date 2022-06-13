import shutil
import pandas as pd
import numpy as np
from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklego.metrics import equal_opportunity_score
from sklego.metrics import p_percent_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold

from sklego.linear_model import DemographicParityClassifier
from sklego.linear_model import EqualOpportunityClassifier
from sklearn.linear_model import LogisticRegression

from moopt import monise

from fair_models import coefficient_of_variation
from fair_models import calc_reweight
from fair_models import FairScalarization, EqualScalarization, EqOpScalarization
from fair_models import SimpleVoting

from model_aggregation import ensemble_filter

# Functions to Minimax
from MMPF.MinimaxParetoFair.MMPF_trainer import SKLearn_Weighted_LLR, APSTAR

# Functions for AdaFair
from AdaFair.AdaFair import AdaFair

# Functions for MAMO-fair
import glob
import torch
import torch.optim as optim
from MAMOfair.dataloader.fairness_datahandler import FairnessDataHandler
from MAMOfair.dataloader.fairness_dataset import CustomDataset
from MAMOfair.public_experiments.pareto_utils import *
from MAMOfair.models.nn1 import NN1
from MAMOfair.loss.losses import *
from MAMOfair.metric.metrics import *
from MAMOfair.trainer import Trainer
from MAMOfair.validator import Validator

def evaluate_model_test(model__, fair_feature, X_test, y_test):
    y_pred = model__.predict(X_test)
    metrics = {"Acc": accuracy_score(y_test, y_pred),
                "BalancedAcc": balanced_accuracy_score(y_test, model__.predict(X_test)),
                "F-score": f1_score(y_test, model__.predict(X_test)),
                "EO": equal_opportunity_score(sensitive_column=fair_feature)(model__, X_test, y_test),
                "DP": p_percent_score(sensitive_column=fair_feature)(model__,X_test),
                "CV": coefficient_of_variation(model__, X_test, y_test)}
    metrics["SingleClass"] = False
    if len(np.unique(y_pred))==1:
        #raise Exception("Model classifies every point to the same class") 
        metrics["SingleClass"] = True
    return metrics


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

def evaluate_adafair(fair_feature, X_train, y_train, X_test, y_test):
    sa_index = list(X_train.columns).index(fair_feature)
    # Train
    adafair_model = AdaFair(n_estimators=300, saIndex=sa_index, saValue=0, c=1)
    adafair_model.fit(X_train, y_train)

    # Evaluate
    adafair_model = evaluate_model_test(adafair_model, fair_feature, X_test, y_test)
    adafair_model['Approach'] = 'AdaFair'

    return adafair_model

def evaluate_mamofair(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    # Process data

    def get_X_y(df, y_cols, keep_sen=True):
        y_rev = y_cols.copy()
        y_rev.reverse()
        col_order = [col for col in df.columns if col not in y_cols] + y_rev
        df = df[col_order]
        y = df[y_cols].to_numpy()
        if(keep_sen is True):
            X = df.drop(y_cols[0], axis=1).to_numpy()
        elif(keep_sen is False):
            X = df.drop(y_cols, axis=1).to_numpy()
        index = {}
        for i, col in enumerate(y_cols):
            index[col] = i
        
        return(X, y, index)

    device = torch.device('cuda:0')

    # Train data
    y_train = y_train.values
    y_train = (y_train+1)/2
    X_train, y, _ = get_X_y(X_train, [fair_feature])
    y_train = np.c_[y_train, y] 
    X1 = torch.from_numpy(X_train).float().to(device)
    y1 = torch.from_numpy(y_train).float().to(device)
    train_data = CustomDataset(X1, y1)

    input_dim = X1.shape[1]

    # Val data
    y_val = y_val.values
    y_val = (y_val+1)/2
    X_val, y, _ = get_X_y(X_val, [fair_feature])
    y_val = np.c_[y_val, y] 
    X1 = torch.from_numpy(X_val).float().to(device)
    y1 = torch.from_numpy(y_val).float().to(device)
    val_data = CustomDataset(X1, y1)

    # Test data
    y_test = y_test.values
    y_test = (y_test+1)/2
    X_test, y, _ = get_X_y(X_test, [fair_feature])
    y_test = np.c_[y_test, y] 
    X1 = torch.from_numpy(X_test).float().to(device)
    y1 = torch.from_numpy(y_test).float().to(device)
    test_data = CustomDataset(X1, y1)

    accuracy = Accuracy(name='accuracy')
    baccuracy = BalancedAccuracy(name='bacc')
    f1score = F1Score(name='f1score')
    dp = DemParity(name='DP')
    eo = EqOportunity(name='EO')
    cv = CoVariation(name='CV')
    
    validation_metrics = [accuracy, baccuracy, f1score, eo, dp, cv]

    data_handler = FairnessDataHandler('data', train_data, val_data, test_data)

    # Build model
    model = NN1(input_dimension=input_dim)
    model.to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    performance_loss = BCELoss(name='bce')
    loss_EOP = TPRLoss(name='EOP', reg_lambda=0.1, reg_type='tanh')
    loss_DDP = DPLoss(name='DPP', reg_lambda=0.1, reg_type='tanh')

    losses = [performance_loss, loss_DDP, loss_EOP]

    save_to_path = 'MAMOfair/saved_models/model/'
    shutil.rmtree(save_to_path, ignore_errors=True)

    trainer = Trainer(data_handler, model, losses, validation_metrics, save_to_path,                      
                            params='MAMOfair/yaml_files/trainer_params.yaml', optimizer=optimizer)

    trainer.train()

    scores_val = to_np(trainer.pareto_manager._pareto_front)
    chosen_score_zenith, idx_zenith = get_solution(scores_val)

    ####### closest to zenith point #############
    model_val = NN1(input_dimension=input_dim)
    model_val.to(device)

    match_zenith = '_'.join(['%.4f']*len(chosen_score_zenith)) % tuple(chosen_score_zenith)
    files = glob.glob(save_to_path + '*')
    for f in files:
        if(match_zenith in f):
            model_val.load_state_dict(torch.load(f))
            continue

    shutil.rmtree(save_to_path, ignore_errors=True)


    # Evaluate
    test_len = data_handler.get_testdata_len()
    test_loader = data_handler.get_test_dataloader(drop_last=False, batch_size=test_len)
    test_validator = Validator(model_val, test_loader, validation_metrics, losses)
    test_metrics, test_losses = test_validator.evaluate()

    mamofair_model = {"Acc": test_metrics[0],
                      "BalancedAcc": test_metrics[1],
                      "F-score": test_metrics[2],
                      "EO": test_metrics[3],
                      "DP": test_metrics[4],
                      "CV": test_metrics[5],
                      "Approach": 'MAMOFair'
    }

    return mamofair_model

def evaluate_mooerr(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test, metric='EO'):
    # Train
    ## Train 150 models
    moo_err = monise(weightedScalar=FairScalarization(X_train, y_train, fair_feature),
                    singleScalar=FairScalarization(X_train, y_train, fair_feature),
                    nodeTimeLimit=2, targetSize=300,
                    targetGap=0, nodeGap=0.05, norm=False)

    moo_err.optimize()

    ## Evaluate the models in val
    mooerr_values = []
    mooerr_sols = []

    for solution in moo_err.solutionsList:
        evaluate = evaluate_model_test(solution.x, fair_feature, X_val, y_val)
        if evaluate['SingleClass']:
            continue
        mooerr_sols.append(solution.x)
        mooerr_values.append(evaluate)

    mooerr_df = pd.DataFrame(mooerr_values)

    mooerr_metrics = ensemble_filter(mooerr_df, mooerr_sols, fair_feature, 
                                    X_test, y_test, n_acc = 150, nds = True, with_acc=False, n_selected=10)

    mooerr_metrics['Approach'] = 'MooErr'
    del mooerr_metrics['Filter']

    clear_output(wait=True)

    return mooerr_metrics

def evaluate_mooacep(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    # Train 150 models
    mooacep = monise(weightedScalar=EqualScalarization(X_train, y_train, fair_feature),
                singleScalar=EqualScalarization(X_train, y_train, fair_feature),
                nodeTimeLimit=2, targetSize=300,
                targetGap=0, nodeGap=0.01, norm=False)

    mooacep.optimize()

    # Evaluate the models
    mooacep_values_val = []
    mooacep_sols = []

    for solution in mooacep.solutionsList:
        evaluate = evaluate_model_test(solution.x, fair_feature, X_val, y_val)
        if evaluate['SingleClass']:
            continue 
        mooacep_sols.append(solution.x)
        mooacep_values_val.append(evaluate)

    mooacep_df = pd.DataFrame(mooacep_values_val)

    mooacep_metrics = ensemble_filter(mooacep_df, mooacep_sols, fair_feature, 
                                    X_test, y_test, n_acc = 150, nds = True, with_acc=False, n_selected=10)

    mooacep_metrics['Approach'] = 'MooAcep'
    del mooacep_metrics['Filter']

    clear_output(wait=True)

    return mooacep_metrics

def evaluate_mooeo(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    # Train 150 models
    mooeo = monise(weightedScalar=EqOpScalarization(X_train, y_train, fair_feature),
                singleScalar=EqOpScalarization(X_train, y_train, fair_feature),
                nodeTimeLimit=2, targetSize=300,
                targetGap=0, nodeGap=0.01, norm=False)

    mooeo.optimize()

    # Evaluate the models
    mooeo_values_val = []
    mooeo_sols = []

    for solution in mooeo.solutionsList:
        evaluate = evaluate_model_test(solution.x, fair_feature, X_val, y_val)
        if evaluate['SingleClass']:
            continue
        mooeo_sols.append(solution.x)
        mooeo_values_val.append(evaluate)

    mooeo_df = pd.DataFrame(mooeo_values_val)

    mooeo_metrics = ensemble_filter(mooeo_df, mooeo_sols, fair_feature, 
                                    X_test, y_test, n_acc = 150, nds = True, with_acc=False, n_selected=10)

    mooeo_metrics['Approach'] = 'MooEO'
    del mooeo_metrics['Filter']

    clear_output(wait=True)

    return mooeo_metrics

def evaluate_all_approaches(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test):
    models_metrics = [evaluate_logreg(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_reweigh(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_dempar(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_eqop(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_adafair(fair_feature, X_train, y_train, X_test, y_test),
                    evaluate_minimax(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test),
                    evaluate_mamofair(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test),
                    evaluate_mooerr(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test),
                    evaluate_mooacep(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test),
                    evaluate_mooeo(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test)
                    ]

    return pd.DataFrame(models_metrics).set_index('Approach')

def kfold_methods(X, y, X_test, y_test, fair_feature, n_folds = 5):
    results_test = pd.DataFrame()

    kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
    kf.get_n_splits(X)

    idx_fold = 0
    for train_index, tv_index in kf.split(X):
        idx_fold += 1
        print(f"  [INFO] Starting Fold {idx_fold}...")

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[tv_index], y.iloc[tv_index]

        eval_result = evaluate_all_approaches(fair_feature, X_train, y_train, X_val, y_val, X_test, y_test)

        results_test = pd.concat((results_test, eval_result))

    return results_test