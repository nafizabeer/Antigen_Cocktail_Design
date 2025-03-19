import os
import multiprocessing as mp
import numpy as np
from functools import partial
import torch

from AMcode_utils.Mixture_simulation import sim_mixture
from BayOpt.cocktail_efficacy import cocktail_efficacy
from BayOpt.comb_bayopt import run_one_replication




import argparse

parser = argparse.ArgumentParser(description='Cocktail design Bayesian Optimization')

parser.add_argument('--data_dir', help='the path to root dir')
parser.add_argument('--out_dir', help='the path to save the best performing model on validation split')


parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--iterations', default=60, type=int, metavar='N',
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

    workers = args.workers


    problem_function = cocktail_efficacy(20, partial(estimate_titer, N = workers, antigen_list = antigen_list, testpanel = testpanel))

    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)
    seed = args.seed
    
    np.random.seed(seed)
    n_initial_points = 5
    X = np.zeros((n_initial_points,20))

    for idx in range(n_initial_points):
      sel_idx = np.random.choice(list(range(20)),size = (3,), replace = False)
      X[idx, sel_idx] = 1

    X = torch.tensor(X)

    kwargs = {"iterations": args.iterations, 
              "batch_size": 1,
              "mc_samples" : 128, #not used 
              "optimization_kwargs": {}, 
              "model_kwargs": {"kernel_type" : 'iso_combo'},
              "X_init": X,
            }


    label = 'pr__ei'
    save_frequency = 1
    filename = os.path.join(output_path, f"{str(seed).zfill(4)}_{label}.pt")
    save_callback = lambda data: torch.save(data, filename)
    filename_2 = os.path.join(output_path, f"{str(seed).zfill(4)}_BO_{label}.npz")
    save_callback_2 = lambda data: np.savez_compressed(filename_2, np.array(data, dtype = object) )

    run_one_replication(
            base_function=problem_function,
            seed=seed,
            label=label,
            save_callback=save_callback,
            save_callback_2 = save_callback_2,
            save_frequency=save_frequency,
            **kwargs,
        )
