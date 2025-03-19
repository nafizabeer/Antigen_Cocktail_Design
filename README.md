# Antigen_Cocktail_Design
Companion repository for manuscript titled "In Silico Design of Immunogenic Antigen Cocktail via Affinity Maturation Guided Optimization"

# Install dependencies

```
conda env create -f env-basic.yml
source activate vd_adv
cd affinity_maturation
pip install -e .
```

# Optimization Experiment
Greedy optimization
```
python greedy_opt_script.py --data_dir "./data" --out_dir "./results" --workers 10
```
Bayesian optimization
```
python bay_opt_script.py --data_dir "./data" --out_dir "./results" --seed 1 --iterations 55 --workers 10
```

Genetic algorithm
```
python opt_script.py --data_dir "./data" --out_dir "./results" --seed 1 --iterations 20 --workers 10
```

# Credits
The codes for Bayesian optimization and genetic algorithm are based on [https://github.com/facebookresearch/bo_pr](https://github.com/facebookresearch/bo_pr) and [https://github.com/100/Solid/blob/master/Solid/GeneticAlgorithm.py](https://github.com/100/Solid/blob/master/Solid/GeneticAlgorithm.py) respectively.

The script for affnity maturation simulation is built on the resouces shared in [https://github.com/ericzwang/sars2-vaccine/tree/main/AMcode](https://github.com/ericzwang/sars2-vaccine/tree/main/AMcode).


