# %%
import experiment_utils
import pandas as pd
import pickle
from pathlib import Path

# %% [markdown]
# # COMPAS Dataset

# %%
dataset = 'compas'
fair_feature = 'race'

X = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y.pickle", 'rb') as f:
    y = pickle.load(f)

X_ftest = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X_ftest.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y_ftest.pickle", 'rb') as f:
    y_ftest = pickle.load(f)

# %%
kfold_results_compas = experiment_utils.kfold_methods(X, y, X_ftest, y_ftest, 
                                                        fair_feature, n_folds = 5)

# %%
results_mean_compas = kfold_results_compas.groupby('Approach').mean()
Path("results/exp_comparative").mkdir(parents=True, exist_ok=True)
results_mean_compas.to_csv('results/exp_comparative/compas.csv')
print(results_mean_compas.to_latex())

# %%
results_mean_compas

# %% [markdown]
# # German Dataset

# %%
dataset = 'german'
fair_feature = 'sex'

X = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y.pickle", 'rb') as f:
    y = pickle.load(f)

X_ftest = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X_ftest.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y_ftest.pickle", 'rb') as f:
    y_ftest = pickle.load(f)

# %%
kfold_results = experiment_utils.kfold_methods(X, y, X_ftest, y_ftest, 
                                                fair_feature, n_folds = 5)

# %%
german_results_mean = kfold_results.groupby('Approach').mean()

german_results_mean.to_csv('results/exp_comparative/german.csv')
print(german_results_mean.to_latex())

# %%
german_results_mean

# %% [markdown]
# # Adult Dataset

# %%
dataset = 'adult'
fair_feature = 'race'

X = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y.pickle", 'rb') as f:
    y = pickle.load(f)

X_ftest = pd.read_pickle("data/"+dataset+"_"+fair_feature+"/X_ftest.pickle")
with open("data/"+dataset+"_"+fair_feature+"/y_ftest.pickle", 'rb') as f:
    y_ftest = pickle.load(f)

# %%
kfold_results = experiment_utils.kfold_methods(X, y, X_ftest, y_ftest, 
                                                fair_feature, n_folds = 5)

# %%
adult_results_mean = kfold_results.groupby('Approach').mean()

# %%
adult_results_mean

# %%
adult_results_mean.to_csv('results/exp_comparative/adult.csv')
print(adult_results_mean.to_latex())

# %%



