#! /usr/bin/bash
SECONDS=0
seed_array=(1 2 3 4 5 )
echo 'Cocktail Design Bayesian Optimization'
for seed in "${seed_array[@]}"; do
    echo "seed: ${seed}"
    python bay_opt_script.py --data_dir "./data" --out_dir "./results" --seed ${seed} --iterations 55 --workers 10
done
echo 'completed'
echo "Elapsed Time: $SECONDS seconds"