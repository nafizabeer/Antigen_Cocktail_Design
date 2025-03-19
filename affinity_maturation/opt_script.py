import os
import multiprocessing as mp
import numpy as np
from functools import partial

from AMcode_utils.Mixture_simulation import sim_mixture
from comb_opt.GeneticAlgorithm import GeneticAlgorithm
from comb_opt.GeneticAlgorithm_v2 import GeneticAlgorithm_v2

import argparse

parser = argparse.ArgumentParser(description='Cocktail design Genetic algorithm')

parser.add_argument('--data_dir', help='the path to data dir')
parser.add_argument('--out_dir', help='the path to save stats')


parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--iterations', default=20, type=int, metavar='N',
                    help='number of total itearations to run')
parser.add_argument('--seed', default=0, type=int, metavar='N',
                    help='random seed')

args = parser.parse_args()


def estimate_titer(sel_vec,N, antigen_list, testpanel):

    with mp.Pool(processes=N) as p:
        results = p.map(partial(sim_mixture, sel_vec = sel_vec, antigen_list = antigen_list, testpanel = testpanel), np.arange(0,N))

    mean_titer = np.hstack([x[1] for x in results])
    # print(f'mean titer: {np.mean(mean_titer)} ({np.std(mean_titer)})')
    
    return np.mean(mean_titer)


if __name__ =='__main__':

    data = np.load(os.path.join(args.data_dir,'Ags_for_AM.npz'))

    antigen_list = data['cand_Ags']
    testpanel = data['test_panel_Ags']
    seed = args.seed
    mut_rate = 0.3
    label = f'mut_rate_{int(mut_rate*100)}'
    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{str(seed).zfill(4)}_GA_{label}.npz")

    save_callback = lambda data: np.savez_compressed(filename, np.array(data, dtype = object) )

    workers = args.workers
    score_func = partial(estimate_titer, N = args.workers, antigen_list = antigen_list, testpanel = testpanel)

    model = GeneticAlgorithm_v2(pop_size = 5, n_genes = 20,selection_rate = 0.5, 
                            mutation_rate = mut_rate, max_steps=args.iterations, 
                            score_func = score_func,
                            max_fitness=None)

    best_solution, best_objective_value = model.run(seed = seed, save_callback = save_callback)
