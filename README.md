# moopt_fairness


## Dependencies

Os modelos implementados nesse projeto utilizam da abordagem [MONISE - Many Objective Non-Inferior Set Estimation](https://www.sciencedirect.com/science/article/abs/pii/S0377221719309282) para otimizar os problemas multi-objetivos gerados. Portanto, é necessário instalar seu código. Clone o repositório https://github.com/marcosmrai/moopt e rode o seguinte comando:

```
python setup.py install 
```

Nos experimentos são feitas comparações com diversos modelos, dentre eles o [Minimax Pareto Fairness](http://proceedings.mlr.press/v119/martinez20a.html). Para isso, clone o seguinte repositório no diretório deste projeto: https://github.com/natalialmg/MMPF