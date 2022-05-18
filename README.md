# moopt_fairness


## Instalation

The `requirements.txt` file list all Python libraries used on this project, which can be installed using:

```
pip install -r requirements.txt
```

### Optimization Algorithm

The optimization algorithm used in our approach is [MONISE - Many Objective Non-Inferior Set Estimation](https://www.sciencedirect.com/science/article/abs/pii/S0377221719309282). To use it, clone the repository https://github.com/marcosmrai/moopt and install it:

```
python setup.py install 
```

If you have any problems with PuLP, install version 2.1:

```
pip install -Iv PuLP==2.1
```

### Compared Approaches

In the experiments, we made comparisons with several approaches. Please clone the following repositories in the directory of this project:

- [Minimax Pareto Fairness](http://proceedings.mlr.press/v119/martinez20a.html): https://github.com/natalialmg/MMPF
- [AdaFair](https://dl.acm.org/citation.cfm?id=3357974): https://github.com/iosifidisvasileios/AdaFair

To use Minimax Pareto Fairness change line 9 of the file `MMPF/dataset_loaders.py` to: `from MMPF.MinimaxParetoFair import *`.

AdaFair uses a different version of sklearn. To solve compatiblity issues, replace line 40  of the `AdaFair.py` file with:

```
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from numpy import float32 as DTYPE
```