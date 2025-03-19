#! /usr/bin/bash
SECONDS=0
seed_array=(1 2 3 4 5 )
echo 'Cocktail Design Genetic Algorithm'
for seed in "${seed_array[@]}"; do
    echo "seed: ${seed}"
    python opt_script.py --data_dir "./data" --out_dir "./results" --seed ${seed} --iterations 20 --workers 10
done
echo 'completed'
echo "Elapsed Time: $SECONDS seconds"