#https://github.com/100/Solid/blob/master/Solid/GeneticAlgorithm.py


from abc import ABCMeta, abstractmethod
from copy import deepcopy
from random import randint, random, shuffle
import numpy as np

key_2_val = lambda key: [int(x) for x in key]
key_idx = lambda p: ''.join([str(int(x)) for x in p])
# fitness function with memory
class scorer_obj():
    def __init__(self, score_func):
        self.score_func = score_func
        self.score_dict = {}
    def score(self, x):
        try:
            return self.score_dict[key_idx(x)]
        except:
            val = self.score_func(x)
            self.score_dict[key_idx(x)] = val
            return val



def get_child_from_crossover(pa1, pa2):
    
    n_genes = len(pa1)
    offspring_size = 2

    pa1 = np.argwhere(pa1).reshape(-1)
    pa2 = np.argwhere(pa2).reshape(-1)

    children = np.zeros((offspring_size, n_genes))
    for idx, x in enumerate(range(1,3)):

        pt1 = np.random.choice(pa1, size = int(x), replace=False)
        pt2 = np.random.choice(list(set(pa2).difference(set(pt1))), size = int(len(pa2)-x), replace=False)

        children[idx, pt1] = 1
        children[idx, pt2] = 1
    
    return children



class GeneticAlgorithm_v2:
    """
    Conducts genetic algorithm
    """
    __metaclass__ = ABCMeta

    population = None
    fitnesses = None

    crossover_rate = None

    mutation_rate = None

    cur_steps = None
    best_fitness = None
    best_member = None

    max_steps = None
    max_fitness = None

    def __init__(self,pop_size, n_genes, selection_rate, mutation_rate, max_steps, score_func,max_fitness=None):
        """
        :param crossover_rate: probability of crossover
        :param mutation_rate: probability of mutation
        :param max_steps: maximum steps to run genetic algorithm for
        :param max_fitness: fitness value to stop algorithm once reached
        """
        self.scorer = scorer_obj(score_func)
        self.pop_size = pop_size
        self.n_genes = n_genes
        if isinstance(selection_rate, float):
            if 0 <= selection_rate <= 1:
                self.selection_rate = selection_rate
            else:
                raise ValueError('Selection rate must be a float between 0 and 1')
        else:
            raise ValueError('Selection rate must be a float between 0 and 1')

        if isinstance(mutation_rate, float):
            if 0 <= mutation_rate <= 1:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('Mutation rate must be a float between 0 and 1')
        else:
            raise ValueError('Mutation rate must be a float between 0 and 1')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError('Maximum steps must be a positive integer')

        if max_fitness is not None:
            if isinstance(max_fitness, (int, float)):
                self.max_fitness = float(max_fitness)
            else:
                raise ValueError('Maximum fitness must be a numeric type')

    def __str__(self):
        return (f'GENETIC ALGORITHM: {len(self.scorer.score_dict)} queries\n CURRENT STEPS: {self.cur_steps} \n BEST FITNESS: { self.best_fitness} \n BEST MEMBER: {str(np.argwhere(self.best_member))} \n\n') 

    

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        :return: None
        """
        self.cur_steps = 0
        self.population = None
        self.fitnesses = None
        self.best_member = None
        self.best_fitness = None

    
    def get_initial_popultation(self,pop_size):
          P = np.zeros((pop_size, self.n_genes))
          
          for idx in range(pop_size):
              sel_idx = np.random.choice(list(range(self.n_genes)),size = (3,), replace = False)
              P[idx, sel_idx] = 1
        
        
          self.population = P

    @abstractmethod
    def _fitness(self, member):
        """
        Evaluates fitness of a given member
        :param member: a member
        :return: fitness of member
        """
        pass

    def _populate_fitness(self):
        """
        Calculates fitness of all members of current population
        :return: None
        """
        self.fitnesses = np.array([self.scorer.score(x) for x in self.population])
        sorted_idx = np.argsort(self.fitnesses)
        self.fitnesses = self.fitnesses[sorted_idx]
        self.population = self.population[sorted_idx]
        

    def _most_fit(self):
        """
        Finds most fit member of current population
        :return: most fit member and most fit member's fitness
        """
        best_idx = np.argmax(self.fitnesses)

        return self.population[best_idx], self.fitnesses[best_idx]

    def _select_n(self, n):
        """
        Probabilistically selects n members from current population using
        roulette-wheel selection
        :param n: number of members to select
        :return: n members
        """
        shuffle(self.population)
        total_fitness = sum(self.fitnesses)
        if total_fitness != 0:
            probs = list([self._fitness(x) / total_fitness for x in self.population])
        else:
            return self.population[0:n]
        res = []
        for _ in range(n):
            r = random()
            sum_ = 0
            for i, x in enumerate(probs):
                sum_ += probs[i]
                if r <= sum_:
                    res.append(deepcopy(self.population[i]))
                    break
        return res

    def select_parent(self,selection_strategy, n_matings):

        # sorted_idx = np.argsort(self.fitnesses)
        # self.fitnesses = self.fitnesses[sorted_idx]
        # self.population = self.population[sorted_idx]

        if selection_strategy == "roulette_wheel":
            prob = np.arange(len(self.population))[::-1] + 1
            prob = prob/prob.sum()
        else:
            prob = np.ones(len(self.population))/len(self.population)

        pa = np.vstack([np.random.choice(len(self.population),size = 2, replace = False, p = prob) for _ in range(n_matings)])

        return pa[:,0], pa[:,1]




    def perform_mutation(self):

        P = deepcopy(self.population)
        
        for x in P:
            if np.random.rand()<self.mutation_rate:
                idx = np.argwhere(x).reshape(-1)
                mut_idx = np.random.choice(idx, size = 1)
                x[mut_idx] = 0
                new_idx = np.random.choice(list(set(range(len(x))).difference(set(idx))), size = 1)
                x[new_idx] = 1

        self.population = P


    def prune_population(self, n_con):

        self.population = self.population[(self.population.sum(axis=-1)>0) & (self.population.sum(axis=-1)<=n_con)]

    
    

    def keep_fixed_population_size(self):

        if len(self.population)> self.pop_size:
            key_vals = [key_idx(x) for x in self.population]
            unique_members = np.unique(key_vals)
            unique_scores = np.array([self.scorer.score_dict[x] for x in unique_members])
            idx = np.argsort(-unique_scores)
            if len(unique_members)>=self.pop_size:
                self.population = unique_members[idx[:self.pop_size]]
            else:
                self.population = np.hstack((unique_members,np.random.choice(unique_members,self.pop_size-len(unique_members),replace=True)))

            self.population = np.array([key_2_val(x) for x in self.population])




    def run(self, seed = 0, verbose=True, save_callback = None):
        """
        Conducts genetic algorithm
        :param verbose: indicates whether or not to print progress regularly
        :return: best state and best objective function value
        """
        self._clear()
        np.random.seed(seed)
        self.get_initial_popultation(self.pop_size)
        self._populate_fitness()
        # print(self.population)
        self.best_member, self.best_fitness = self._most_fit()
        best_stats = {'score' : [self.best_fitness], 'best_member' : [self.best_member], 'num_queries': [len(self.scorer.score_dict)]}
        print(self)

        n_matings = np.maximum(int(self.pop_size*(1-self.selection_rate)/2),1)
        
        
        for i in range(self.max_steps):
            self.cur_steps += 1


            parents = self.select_parent(selection_strategy = 'roulette_wheel', n_matings = n_matings)

            P_new = []
            for pa in zip(parents[0],parents[1]):
                P_new.append(get_child_from_crossover(self.population[pa[0]], self.population[pa[1]]))

            self.population = np.vstack((self.population,np.vstack(P_new)))
            
            self.perform_mutation()

            self.prune_population(n_con = 3)
            self._populate_fitness()
            self.keep_fixed_population_size()
            self._populate_fitness()

            best_member, best_fitness = self._most_fit()

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_member = deepcopy(best_member)
                best_stats['num_queries'].append(len(self.scorer.score_dict))
            else:
                best_stats['num_queries'].append(best_stats['num_queries'][-1])

            if self.max_fitness is not None and self.best_fitness >= self.max_fitness:
                print("TERMINATING - REACHED MAXIMUM FITNESS")
                return self.best_member, self.best_fitness
                
            best_stats['score'].append(self.best_fitness)
            best_stats['best_member'].append(self.best_member)
            
            if verbose and ((i + 1) % 1 == 0):
                print(self)

            if save_callback is not None:
                save_callback(best_stats)
        # np.savez('score_obj.npz', np.array(self.scorer.score_dict, dtype = object))
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best_member, self.best_fitness