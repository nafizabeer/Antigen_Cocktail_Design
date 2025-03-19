import os
import multiprocessing as mp
import numpy as np
from functools import partial

from AMcode_utils.Mixture_simulation import sim_mixture

import argparse

parser = argparse.ArgumentParser(description='Cocktail design greedy optimization')

parser.add_argument('--data_dir', help='the path to data dir')
parser.add_argument('--out_dir', help='the path to save stats')


parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

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
    
    
    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"greedy.npz")

    save_callback = lambda data: np.savez_compressed(filename, np.array(data, dtype = object) )

    workers = args.workers
    score_func = partial(estimate_titer, N = args.workers, antigen_list = antigen_list, testpanel = testpanel)


    best_solution = []
    Ags_list = set(range(20))
    results = {'cocktail':[], 'score':[]}
    for _ in range(3):
        efficacy_list = []
        for Ag_idx in Ags_list:
            sel_vec = np.zeros(20)
            cocktail = best_solution + [Ag_idx]
            sel_vec[cocktail] = 1
            efficacy_list.append(score_func(sel_vec))
            results['cocktail'].append(cocktail)
            results['score'].append(efficacy_list[-1])

        assert len(efficacy_list)==len(Ags_list)

        sel_idx = np.argmax(efficacy_list)
        best_solution.append(list(Ags_list)[sel_idx])
        print(f"best cocktail of {_+1} Ag: {best_solution} with efficacy {efficacy_list[sel_idx]}")
        Ags_list = Ags_list.difference({list(Ags_list)[sel_idx]})


    save_callback(results)
