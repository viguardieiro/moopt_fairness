from sklearn.metrics import log_loss
from sklearn.utils.extmath import squared_norm
from moopt.scalarization_interface import scalar_interface, single_interface, w_interface
from moopt import monise
from sklearn.linear_model import LogisticRegression
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

class FairScalarization(w_interface, single_interface, scalar_interface):
    def __init__(self, X, y, fair_feature):
        self.fair_feature = fair_feature
        self.fair_att = sorted(X[fair_feature].unique())
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
        
        #sample_weight = self.X[self.fair_feature].replace({ff:fw for ff, fw in zip(self.fair_att,fair_weight)})
        sample_weight = self.X[self.fair_feature].replace({ff:fw/sum(self.X[self.fair_feature]==ff) for ff, fw in zip(self.fair_att,fair_weight)})
        prec = np.mean(sample_weight)
        reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=None,
                                 penalty='l2', max_iter=10**4, tol=prec*10**-6, 
                                 C=1/lambd).fit(self.X, self.y, sample_weight=sample_weight.values)
        
        y_pred = reg.predict_proba(self.X)
        
        self.__objs = np.zeros(len(self.fair_att)+1)
        for i, feat in enumerate(self.fair_att):
            fair_weight = np.zeros(len(self.fair_att))
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feature].replace({ff:fw for ff, fw in zip(self.fair_att,fair_weight)})
            #self.__objs[i] = log_loss(self.y, y_pred, sample_weight=sample_weight)*sum(self.X[self.fair_feature]==feat)
            self.__objs[i] = log_loss(self.y, y_pred, sample_weight=sample_weight)
        
        self.__objs[-1] = squared_norm(reg.coef_)
        self.__x = reg
        return self

class MOOLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, metric='accuracy'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        self.moo_ = None

    def tune(self, metric=None):
        self.best_perf = 0
        self.best_model = None

        if metric is not None:
            self.metric = metric
        if self.moo_ is None:
            self.moo_ = monise(weightedScalar=FairScalarization(self.X_train, self.y_train, 'Sex'), singleScalar=FairScalarization(self.X_train, self.y_train, 'Sex'),
                          nodeTimeLimit=2, targetSize=150,
                          targetGap=0, nodeGap=0.01, norm=False)
            self.moo_.optimize()
        for solution in self.moo_.solutionsList:
            y_pred = solution.x.predict(self.X_val)
            
            if (sklearn.metrics.accuracy_score(self.y_val, y_pred)==0 or
                equal_opportunity_score(sensitive_column="Sex")(solution.x, self.X_val, self.y_val)==0 or
                p_percent_score(sensitive_column="Sex")(solution.x, self.X_val))==0:
                continue
            
            if self.metric=='accuracy':
                perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
            elif self.metric=='equal_opportunity':
                perf = equal_opportunity_score(sensitive_column="Sex")(solution.x, self.X_val, self.y_val)
            elif self.metric=='p_percent':
                perf = p_percent_score(sensitive_column="Sex")(solution.x, self.X_val)
            elif self.metric=='c_variation':
                perf = 1/coefficient_of_variation(solution.x, self.X_val, self.y_val)
            
            if perf>self.best_perf:
                self.best_perf = perf
                self.best_model = solution.x

        return self.best_model
        
class FindCLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, sample_weight=None, metric='accuracy'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
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
            equal_opportunity_score(sensitive_column="Sex")(model, self.X_val, self.y_val)==0 or
            p_percent_score(sensitive_column="Sex")(model, self.X_val))==0:
            return float('inf')
        
        if self.metric=='accuracy':
            perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
        elif self.metric=='equal_opportunity':
            perf = equal_opportunity_score(sensitive_column="Sex")(model, self.X_val, self.y_val)
        elif self.metric=='p_percent':
            perf = p_percent_score(sensitive_column="Sex")(model, self.X_val)
        elif self.metric=='c_variation':
            perf = 1/coefficient_of_variation(model, self.X_val, self.y_val)
        
        if perf>self.best_perf:
            self.best_perf = perf
            self.best_model = model

        error = 1/perf

        return error  # An objective value linked with the Trial object.
    def tune(self):
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.objective, n_trials=100)
        
        return self.best_model
    
class FindCCLogisticRegression():
    def __init__(self, X_train, y_train, X_val, y_val, sample_weight=None, metric='accuracy', base_model='demografic'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
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
            if self.base_model=='equal':
                model = EqualOpportunityClassifier(sensitive_cols="Sex", positive_target=True, covariance_threshold=c, C=C, max_iter=10**3)
            else:
                model = DemographicParityClassifier(sensitive_cols="Sex", covariance_threshold=c, C=C, max_iter=10**3)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)
        except:
            return float('inf')


        
        if (sklearn.metrics.accuracy_score(self.y_val, y_pred)==0 or
            equal_opportunity_score(sensitive_column="Sex")(model, self.X_val, self.y_val)==0 or
            p_percent_score(sensitive_column="Sex")(model, self.X_val))==0:
            return float('inf')

        
        if self.metric=='accuracy':
            perf = sklearn.metrics.accuracy_score(self.y_val, y_pred)
        elif self.metric=='equal_opportunity':
            perf = equal_opportunity_score(sensitive_column="Sex")(model, self.X_val, self.y_val)
        elif self.metric=='p_percent':
            perf = p_percent_score(sensitive_column="Sex")(model, self.X_val)
        elif self.metric=='c_variation':
            perf = 1/coefficient_of_variation(model, self.X_val, self.y_val)
        
        if perf>self.best_perf:
            self.best_perf = perf
            self.best_model = model
        
        error = 1/perf

        return error  # An objective value linked with the Trial object.
    def tune(self):
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study()  # Create a new study.
        study.optimize(self.objective, n_trials=100)
        
        return self.best_model