from sklearn.metrics import log_loss
from sklearn.utils.extmath import squared_norm
from moopt.scalarization_interface import scalar_interface, single_interface, w_interface
from moopt import monise
import numpy as np
import sklearn
from sklego.metrics import equal_opportunity_score
from sklego.metrics import p_percent_score
from sklearn.metrics import log_loss
from sklearn.utils.extmath import squared_norm
from sklego.linear_model import DemographicParityClassifier
from sklego.linear_model import EqualOpportunityClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import pandas as pd
from scipy import stats
import math

import sys
sys.path.append("./MMFP/")
from MMPF.MinimaxParetoFair.MMPF_trainer import SKLearn_Weighted_LLR, APSTAR

#import DES techniques from DESlib
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES

def calc_reweight(X, y, fair_feat):
    W = {0: {}, 1: {}}
    D = len(X)
    len_g0 = X.groupby(fair_feat).size()[0]
    len_g1 = X.groupby(fair_feat).size()[1]
    len_neg = sum(y==-1)
    len_pos = sum(y==1)
    len_g0_pos = len(X[(X[fair_feat] == 0) & (y == 1)])
    len_g0_neg = len(X[(X[fair_feat] == 0) & (y == -1)])
    len_g1_pos = len(X[(X[fair_feat] == 1) & (y == 1)])
    len_g1_neg = len(X[(X[fair_feat] == 1) & (y == -1)])

    W[0][1] = (len_g0*len_pos)/(D*len_g0_pos)
    W[0][-1] = (len_g0*len_neg)/(D*len_g0_neg)

    W[1][1] = (len_g1*len_pos)/(D*len_g1_pos)
    W[1][-1] = (len_g1*len_neg)/(D*len_g1_neg)

    return [W[X.iloc[i][fair_feat]][y.iloc[i]] for i in range(X.shape[0])]

def generalized_entropy_index(model, X, y_true, alpha=2, target=1):
    y_pred = model.predict(X)

    b = 1 + 1*(y_pred==target) - 1*(y_true==target)
    mi = np.mean(b)

    if alpha == 1:
        return np.mean(np.log((b/mi)**b)/mi)
    elif alpha == 0:
        return -np.mean(np.log(b/mi)/mi)
    else:
        return np.mean((b/mi)**alpha-1)/(alpha*(alpha-1))

def coefficient_of_variation(model, X, y_true, target=1):
    return 2*(generalized_entropy_index(model, X, y_true, alpha=2, target=target)**0.5)

class SimpleVoting():
    def __init__(self, estimators, voting='hard', minimax=False):
        self.estimators = estimators
        self.voting = voting
        if minimax:
            self.classes_ = estimators[0][1].model.classes_
        else:
            self.classes_ = estimators[0][1].classes_
    
    def predict(self, X):
        if self.voting != 'soft':
            return stats.mode([m[1].predict(X) for m in self.estimators],axis=0)[0][0]

        argmax = np.argmax(np.mean([m[1].predict_proba(X) for m in self.estimators],axis=0), axis=1)
        return np.array([self.classes_[v] for v in argmax])
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return sklearn.metrics.accuracy_score(y, y_pred)

class FairScalarization(w_interface, single_interface, scalar_interface):
    def __init__(self, X, y, fair_feat):
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = len(self.fair_att)+1
        self.X, self.y = X, y

    @property
    def M(self):
        return self.__M

    @property
    def feasible(self):
        return True

    @property
    def optimum(self):
        return True

    @property
    def objs(self):
        return self.__objs

    @property
    def x(self):
        return self.__x

    @property
    def w(self):
        return self.__w

    def optimize(self, w):
        """Calculates the a multiobjective scalarization"""
        if type(w) is int:
            self.__w = np.zeros(self.M)
            self.__w[w] = 1
        elif type(w) is np.ndarray and w.ndim==1 and w.size==self.M:
            self.__w = w
        else:
            raise('w is in the wrong format')
        #print('w', self.__w)

        if self.__w[-1]==0:
            lambd=10**-20
        elif self.__w[-1]==1:
            lambd=10**20
        else:
            lambd = self.__w[-1]/(1-self.__w[-1])
        fair_weight = self.__w[:-1]*(1+lambd)
        
        sample_weight = self.X[self.fair_feat].replace({ff:fw/sum(self.X[self.fair_feat]==ff) for ff, fw in zip(self.fair_att,fair_weight)})
        prec = np.mean(sample_weight)
        reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=None,
                                 penalty='l2', max_iter=10**4, tol=prec*10**-6, 
                                 C=1/lambd).fit(self.X, self.y, sample_weight=sample_weight.values)
        
        y_pred = reg.predict_proba(self.X)
        
        self.__objs = np.zeros(len(self.fair_att)+1)
        for i, feat in enumerate(self.fair_att):
            fair_weight = np.zeros(len(self.fair_att))
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feat].replace({ff:fw for ff, fw in zip(self.fair_att,fair_weight)})
            self.__objs[i] = log_loss(self.y, y_pred, sample_weight=sample_weight)
        
        self.__objs[-1] = squared_norm(reg.coef_)
        self.__x = reg
        return self
    
class EqualScalarization(w_interface, single_interface, scalar_interface):
    def __init__(self, X, y, fair_feat):
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = len(self.fair_att)+2
        self.N = X.shape[0]
        self.X = X.append(X)
        self.y = y.append(pd.Series(np.ones(self.N)))

    @property
    def M(self):
        return self.__M

    @property
    def feasible(self):
        return True

    @property
    def optimum(self):
        return True

    @property
    def objs(self):
        return self.__objs

    @property
    def x(self):
        return self.__x

    @property
    def w(self):
        return self.__w

    def optimize(self, w):
        """Calculates the a multiobjective scalarization"""
        if type(w) is int:
            self.__w = np.zeros(self.M)
            self.__w[w] = 1
        elif type(w) is np.ndarray and w.ndim==1 and w.size==self.M:
            self.__w = w
        else:
            raise('w is in the wrong format')
        #print('w', self.__w)
            
        if self.__w[-1]==0:
            lambd=10**-20
        elif self.__w[-1]==1:
            lambd=10**20
        else:
            lambd = self.__w[-1]/(1-self.__w[-1])
        
        loss_weight = self.__w[0]*(1+lambd)
        equal_weight = self.__w[1:-1]*(1+lambd)
        
        sample_weight = self.X[self.fair_feat].replace({ff:fw for ff, fw in zip(self.fair_att,equal_weight)})
        sample_weight[:self.N] = loss_weight
        self.sample_weight = sample_weight
        prec = np.mean(sample_weight)
        
        reg = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                 penalty='l2', max_iter=10**4, tol=prec*10**-6, 
                                 C=1/lambd).fit(self.X, self.y, sample_weight=sample_weight)
        
        y_pred = reg.predict_proba(self.X)
        
        self.__objs = np.zeros(self.M)
        self.__objs[0] = log_loss(self.y[:self.N], y_pred[:self.N])*self.N
        
        for i, feat in enumerate(self.fair_att):
            equal_weight = np.zeros(len(self.fair_att))
            equal_weight[i] = 1
            
            sample_weight = self.X[self.fair_feat].replace({ff:fw for ff, fw in zip(self.fair_att,equal_weight)})
            sample_weight[:self.N] = 0
            
            self.__objs[i+1] = log_loss(self.y, y_pred, sample_weight=sample_weight)*sum(self.X[self.fair_feat]==feat)/2
        
        self.__objs[-1] = squared_norm(reg.coef_)
        self.__x = reg
        #print('objs', self.__objs)
        return self

class MOOLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, fair_feat, scalarization, metric='accuracy', ensemble='voting'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.fair_feat = fair_feat
        self.scalarization = scalarization
        self.metric = metric
        self.ensemble = ensemble
        self.moo_ = None
        self.solutions_ = None

    def tune(self, metric=None):
        self.best_perf = 0
        self.best_model = None

        if metric is not None:
            self.metric = metric
        if self.moo_ is None:
            self.moo_ = monise(weightedScalar=self.scalarization, singleScalar=self.scalarization,
                          nodeTimeLimit=2, targetSize=150,
                          targetGap=0, nodeGap=0.01, norm=False)
            self.moo_.optimize()
        for solution in self.moo_.solutionsList:
            y_pred = solution.x.predict(self.X_val)
            
            if (sklearn.metrics.accuracy_score(self.y_val, y_pred)==0 or
                equal_opportunity_score(sensitive_column=self.fair_feat)(solution.x, self.X_val, self.y_val)==0 or
                p_percent_score(sensitive_column=self.fair_feat)(solution.x, self.X_val))==0:
                continue
            
            if self.metric=='accuracy':
                perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
            elif self.metric=='equal_opportunity':
                perf = equal_opportunity_score(sensitive_column=self.fair_feat)(solution.x, self.X_val, self.y_val)
            elif self.metric=='p_percent':
                perf = p_percent_score(sensitive_column=self.fair_feat)(solution.x, self.X_val)
            elif self.metric=='c_variation':
                perf = 1/coefficient_of_variation(solution.x, self.X_val, self.y_val)
            
            if perf>self.best_perf:
                self.best_perf = perf
                self.best_model = solution.x

        return self.best_model
    def ensemble_model(self, ensemble=None):
        if ensemble is not None:
            self.ensemble = ensemble
        if self.moo_ is None:
            self.moo_ = monise(weightedScalar=self.scalarization, 
                               singleScalar=self.scalarization,
                              nodeTimeLimit=2, targetSize=150,
                              targetGap=0, nodeGap=0.01, norm=False)
            self.moo_.optimize()
            self.solutions_ = []

            for solution in self.moo_.solutionsList:
                self.solutions_.append(solution.x)
        if self.solutions_ is None:
            self.solutions_ = []

            for solution in self.moo_.solutionsList:
                self.solutions_.append(solution.x)

        if self.ensemble in ['voting', 'voting hard']:
            models_t = [
                ("Model " + str(i), self.solutions_[i])
                for i in range(len(self.solutions_))
            ]

            ensemble_model = SimpleVoting(estimators=models_t)
        if self.ensemble == 'voting soft':
            models_t = [
                ("Model " + str(i), self.solutions_[i])
                for i in range(len(self.solutions_))
            ]

            ensemble_model = SimpleVoting(estimators=models_t, voting='soft')
        if self.ensemble == 'knorau':
            ensemble_model = KNORAU(self.solutions_)
            ensemble_model.fit(self.X_val, self.y_val)
        if self.ensemble == 'knorae':
            ensemble_model = KNORAE(self.solutions_)
            ensemble_model.fit(self.X_val, self.y_val)

        return ensemble_model
        
class FindCLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, fair_feat, sample_weight=None, metric='accuracy'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.fair_feat = fair_feat
        self.best_perf = 0
        self.best_model = None
        self.sample_weight = sample_weight
        self.metric = metric

    def objective(self, trial):
        C = trial.suggest_loguniform('C', 1e-10, 1e10)
        model = LogisticRegression(C=C, max_iter=10**3, tol=10**-6)

        model.fit(self.X_train, self.y_train, sample_weight=self.sample_weight)
        y_pred = model.predict(self.X_val)

        if (sklearn.metrics.accuracy_score(self.y_val, y_pred)==0 or
            equal_opportunity_score(sensitive_column=self.fair_feat)(model, self.X_val, self.y_val)==0 or
            p_percent_score(sensitive_column=self.fair_feat)(model, self.X_val))==0:
            return float('inf')

        if self.metric=='accuracy':
            perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
        elif self.metric=='equal_opportunity':
            perf = equal_opportunity_score(sensitive_column=self.fair_feat)(model, self.X_val, self.y_val)
        elif self.metric=='p_percent':
            perf = p_percent_score(sensitive_column=self.fair_feat)(model, self.X_val)
        elif self.metric=='c_variation':
            perf = 1/coefficient_of_variation(model, self.X_val, self.y_val)

        if perf>self.best_perf:
            self.best_perf = perf
            self.best_model = model

        return 1/perf if perf!=0 else float('inf')
    def tune(self):
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.objective, n_trials=100)
        
        return self.best_model
    
class FindCCLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, fair_feat, sample_weight=None, metric='accuracy', base_model='demografic'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.fair_feat = fair_feat
        self.best_perf = 0
        self.best_model = None
        self.sample_weight = sample_weight
        self.metric = metric
        self.base_model = base_model

    def objective(self, trial):
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        c = trial.suggest_loguniform('c', 1e-5, 1e5)
        #print(c, C)
        try:
        #if 1==1:
            if self.base_model=='equal':
                model = EqualOpportunityClassifier(sensitive_cols=self.fair_feat, positive_target=True, covariance_threshold=c, C=C, max_iter=10**3)
                model.fit(self.X_train, self.y_train)
            elif self.base_model=='demographic':
                model = DemographicParityClassifier(sensitive_cols=self.fair_feat, covariance_threshold=c, C=C, max_iter=10**3)
                model.fit(self.X_train, self.y_train)
            elif self.base_model=='minimax':
                a_train = self.X_train[self.fair_feat].copy().astype('int')
                a_val = self.X_val[self.fair_feat].copy().astype('int')

                a_train[a_train==0] = -1
                a_val[a_val==0] = -1

                model = SKLearn_Weighted_LLR(self.X_train.values, self.y_train.values,
                             a_train.values, self.X_val.values,
                             self.y_val.values, a_val.values,
                             C_reg=C)

                mua_ini = np.ones(a_val.max() + 1)
                mua_ini /= mua_ini.sum()
                results = APSTAR(model, mua_ini, niter=200, max_patience=200, Kini=1,
                                      Kmin=20, alpha=0.5, verbose=False)
                mu_best_list = results['mu_best_list']

                mu_best = mu_best_list[-1]
                model.weighted_fit(self.X_train.values, self.y_train.values, a_train.values, mu_best)
            else:
                raise('Incorrect base_model.')

            y_pred = model.predict(self.X_val)
        except:
            return float('inf')



        if (sklearn.metrics.accuracy_score(self.y_val, y_pred)==0 or
            equal_opportunity_score(sensitive_column=self.fair_feat)(model, self.X_val, self.y_val)==0 or
            p_percent_score(sensitive_column=self.fair_feat)(model, self.X_val))==0:
            return float('inf')


        if self.metric=='accuracy':
            perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
        elif self.metric=='equal_opportunity':
            perf = equal_opportunity_score(sensitive_column=self.fair_feat)(model, self.X_val, self.y_val)
        elif self.metric=='p_percent':
            perf = p_percent_score(sensitive_column=self.fair_feat)(model, self.X_val)
        elif self.metric=='c_variation':
            perf = 1/coefficient_of_variation(model, self.X_val, self.y_val)

        if perf>self.best_perf:
            self.best_perf = perf
            self.best_model = model

        return 1/perf if perf!=0 else float('inf')
    def tune(self):
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.objective, n_trials=100)
        
        return self.best_model