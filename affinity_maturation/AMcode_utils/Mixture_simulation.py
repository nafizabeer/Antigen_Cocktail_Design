# Adapted from affinity maturation code at https://github.com/vanouk/Affinity-Maturation-PNAS

import csv
import sys
import os
import importlib
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer
import importlib
import gc

# Upload dictionary of antigens
from AMcode import dictionary_little_code
importlib.reload(dictionary_little_code)
from AMcode.dictionary_little_code import dicAgs
from AMcode.dictionary_little_code import dicconc
from AMcode.dictionary_little_code import dicGCDur
from AMcode.dictionary_little_code import flag


# data = np.load('./AMcode/Ag_info/Ags_for_AM.npz')

# antigen_list = data['cand_Ags']
# testpanel = data['test_panel_Ags']
# antigen_list = [AgC6]*6

#Upload seeding cells
from AMcode import seedingBcell
importlib.reload(seedingBcell)
from AMcode.seedingBcell import seedingCells
from AMcode.seedingBcell import residue_probs
from AMcode.seedingBcell import residue_escape

###### Global parameters ######
num_naive_seed    = 0
nb_GC_founders    = 10
nb_trial          = 2
breadth_threshold = 17
length            = 50
consLength        = 9
p_mut        = 0.14                             # probability of mutation per division round
p_CDR        = 1.00                             # probability of mutation in the CDR region
p_CDR_lethal = 0.30                             # probability that a CDR mutation is lethal
p_CDR_silent = 0.50                             # probability that a CDR mutation is silent
p_CDR_affect = 1. - p_CDR_lethal - p_CDR_silent # probability that a CDR mutation affects affinity
p_FR_lethal  = 0.80                             # probability that a framework (FR) mutation is lethal
p_FR_silent  = 0.                               # probability that a FR mutation is silent
p_FR_affect  = 1. - p_FR_lethal - p_FR_silent   # probability that a FR mutation affects affinity
all_Ag_per_cycle  = True                        # all Ag per cycle or one Ag per cycle

epsilon           = 1e-16
testpanelSize     = 100
alpha             = 0.0
activation_energy = 9
dx_mutation       = 1.0
dx_penalty        = 1.0
h_high            = 1.5
h_low             = -1

energy_scale = 0.08            # inverse temperature
maxQ          = 1              # max value for Q
minQ          = 1              # min value for Q
sigmaQ       = 0.00            # standard deviation for changes in flexibility with FR mutation
help_cutoff  = 0.70            # only B cells in the top (help_cutoff) fraction of binders receive T cell help
p_recycle    = 0.70            # probability that a B cell is recycled
p_exit       = 1. - p_recycle  # probability that a B cell exits the GC

mu     =  1.9   # lognormal mean
sigma  =  0.5   # lognormal standard deviation
corr   =  0.0   # correlation between antigen variable regions
o      =  3.0   # lognormal offset
mumat  = mu * np.ones(length)
sigmat = sigma * np.diag(np.ones(length))


def create_test_panel(panelSize):
    np.random.seed(1324)
    varLength=length-consLength
    testPanel = {}
    for i in range(panelSize):
        testAg = [np.random.choice([-1, 1], p=[1-prob, prob]) for prob in residue_probs]
        for j in range(consLength): testAg=np.append(testAg,1).tolist()
        if len(testAg) != length:
            raise ValueError('Lengths dont match')
        testPanel.update({i: testAg})
    return testPanel


# testpanel = create_test_panel(testpanelSize) 
    
###### B cell clone class ######

class BCell:

    def __init__(self, nb = 512, **kwargs):
        """ Initialize clone-specific variables. 
            nb          - population size
            res         - array of the values for each residue of the B cell 
            E           - total binding energy
            Ec          - binding energy to the conserved residues
            Q           - overlap parameter, proxy for rigidity (most 0 ---> 1 least flexible)
            nb_FR_mut   - number of accumulated FR mutations
            nb_CDR_mut  - number of accumulated CDR mutations
            antigens    - list of antigens
            breadth     - breadth against test panel
            nb_Ag       - number of antigens available
            last_bound  - number of individuals that last bound each Ag
            mut_res_id  - index of mutated residue
            delta_res   - incremental change at residue
            generation  - generation in the GC reaction
            cycle_number - cycle number
            picked_Ag   - which Ag is picked if one at a time
            history     - history of mutations, generation occurred, and effect on Q/Ec and breadth against test panel """
           
        self.nb = nb    # default starting population size = 512 (9 rounds of division)
        self.picked_Ag = None
        self.res = np.array(kwargs['res']) if 'res' in kwargs else np.zeros(length)
        self.antigens = np.array(kwargs['antigens']) if 'antigens' in kwargs else np.array([np.ones(length)])       
        self.E = kwargs['E'] if 'E' in kwargs else sum(self.res) #assuming that the initializing Ag equals ones(length)
        self.Ec = kwargs['Ec'] if 'Ec' in kwargs else sum(self.res[i] for i in range(length-consLength, length))
        self.breadth = np.array(kwargs['breadth']) if 'breadth' in kwargs else 0
        self.Q = kwargs['Q'] if 'Q' in kwargs else maxQ
        self.nb_FR_mut = kwargs['nb_FR_mut'] if 'nb_FR_mut' in kwargs else 0
        self.nb_CDR_mut = kwargs['nb_CDR_mut'] if 'nb_CDR_mut' in kwargs else 0   
        self.mut_res_id = kwargs['mut_res_id'] if 'mut_res_id' in kwargs else length
        self.mut_res_id_his = kwargs['mut_res_id_his'] if 'mut_res_id_his' in kwargs else []
        self.delta_e = kwargs['delta_e'] if 'delta_e' in kwargs else 0
        self.delta_res = kwargs['delta_res'] if 'delta_res' in kwargs else 0
        self.generation = kwargs['generation'] if 'generation' in kwargs else 0
        self.cycle_number = kwargs['cycle_number'] if 'cycle_number' in kwargs else 2
        
        if 'nb_Ag' in kwargs:
            self.nb_Ag = kwargs['nb_Ag']
        elif np.shape(self.antigens)[0] == length:
            self.nb_Ag = 1   # assuming that the number of antigens in a cocktail is always smaller than the number of residues
        else:
            self.nb_Ag = np.shape(self.antigens)[0]
        
        if 'last_bound' in kwargs:
            self.last_bound = kwargs['last_bound']
        else:
            self.last_bound = np.random.multinomial(self.nb, pvals = [1/float(self.nb_Ag)]*self.nb_Ag)
                   
        if 'history' in kwargs:
            self.history = kwargs['history']
        else:
            self.history = {'generation' : [self.generation], 'cycle_number' : [self.cycle_number],
                            'res' : [self.res], 'nb_CDR_mut' : [self.nb_CDR_mut],
                            'mut_res_id' : [self.mut_res_id], 'E' : [self.E],
                            'delta_res' : [self.delta_res], 'delta_e': [self.delta_e]}

    """ Return a new copy of the input BCell"""
    @classmethod
    def clone(cls, b):
        return cls(1,
                   res = deepcopy(b.res),
                   antigens = deepcopy(b.antigens),
                   cycle_number = b.cycle_number,
                   nb_Ag = b.nb_Ag,
                   E = b.E,
                   Ec = b.Ec,
                   breadth = b.breadth,
                   Q = b.Q,
                   generation = b.generation,
                   mut_res_id = b.mut_res_id,
                   nb_FR_mut = b.nb_FR_mut,
                   nb_CDR_mut = b.nb_CDR_mut,
                   delta_res = b.delta_res,
                   last_bound = deepcopy(b.last_bound),
                   history = deepcopy(b.history),
                   mut_res_id_his = b.mut_res_id_his,
                   delta_e = b.delta_e,
                   picked_Ag = deepcopy(b.picked_Ag))
                   
    def update_history(self):
        """ Add current parameters to the history list. """
        self.history['generation'].append(self.generation)
        self.history['res'].append(self.res)
        self.history['nb_CDR_mut'].append(self.nb_CDR_mut)
        self.history['mut_res_id'] = self.history['mut_res_id'] + [self.mut_res_id]
        self.history['E'].append(self.E)      
        self.history['delta_res'].append(self.delta_res)
        self.history['cycle_number'].append(self.cycle_number)
        self.history['delta_e'].append(self.delta_e)

    def energy(self, Ag):
        """ Return binding energy with input antigen. """    
        weighted = np.multiply(self.res, residue_escape)
        return np.sum(np.multiply(weighted, Ag))

    def conserved_energy(self):
        """ Return binding energy for conserved residues. """
        return sum(self.res[i]*residue_escape[i] for i in range(length-consLength, length))   

    def divide(self, cycle_number):
        """ Run one round of division. """
        self.nb *= 2
        self.generation += 1
        self.cycle_number = cycle_number

    def pick_Antigen(self):
        """ Assuming nb_Ag > 1, return one antigen randomly chosen. """
        return self.antigens[np.random.randint(self.nb_Ag)]

    def calculate_breadth(self, testpanel, threshold, panelSize):   
        test_energies = [self.energy(testpanel[j]) for j in range(testpanelSize)]
        return float(sum(x > threshold for x in test_energies))/panelSize
        
    def update_antigens(self, newAntigens):
        self.antigens = deepcopy(newAntigens)
        if np.shape(newAntigens)[0] == length: self.nb_Ag = 1
        else:                                  self.nb_Ag = np.shape(newAntigens)[0]
    
    def mutate_CDR(self, Ag): ### change parameter of log normal for variable and conserved residues
        """ Change in energy due to affinity-affecting CDR mutation. Only one residue mutates."""
        temp_res1 = deepcopy(self.res)
        temp_cp   = deepcopy(self.res)
        index = np.random.randint(0, length) #randomly chosen residue to be mutated
        self.mut_res_id = index
        self.mut_res_id_his.extend([index])
        delta = (o - np.exp(np.random.normal(mu, sigma)))
        self.delta_e = delta

        if delta > dx_mutation:
            delta = dx_mutation
        elif delta < -dx_mutation:
            delta = -dx_mutation        
        self.delta_res = delta / (Ag[index]*residue_escape[index])
        temp_res1[index] +=  delta / (Ag[index]*residue_escape[index])
        if temp_res1[index] > h_high:
            temp_res1[index] = h_high
        elif temp_res1[index] < h_low:
            temp_res1[index] = h_low
        self.nb_CDR_mut += 1
        self.Ec = self.conserved_energy()
        # self.breadth = self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
        self.E = self.energy(Ag)
        self.res = deepcopy(temp_res1)        
        # self.update_history()
        
        if (index < length-consLength) and temp_cp[index] != h_high and temp_cp[index] != h_low:
            temp_res2 = deepcopy(self.res)
            indexPenalty = np.random.randint(0, consLength) + (length - consLength)
            self.mut_res_id = indexPenalty 
            deltaPenalty = - alpha * delta
            self.delta_e = deltaPenalty
            if deltaPenalty > dx_penalty:
                deltaPenalty = dx_penalty
            elif deltaPenalty < -dx_penalty:
                deltaPenalty = -dx_penalty    
            temp_res2[indexPenalty] +=  deltaPenalty
            if temp_res2[indexPenalty] > h_high:
                temp_res2[indexPenalty] = h_high
            elif temp_res2[indexPenalty] < h_low:
                temp_res2[indexPenalty] = h_low
            self.delta_res = deltaPenalty
            # self.breadth = self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
            self.Ec = self.conserved_energy()
            self.E = self.energy(Ag)
            self.res = deepcopy(temp_res2)
            self.mut_res_id_his.extend([indexPenalty])
            # self.update_history()

    def mutate_FR(self):
        """ Change in flexibility due to affinity-affecting framework (FR) mutation. """
        dQ = np.random.normal(0, sigmaQ)
        if self.Q + dQ > maxQ:
            self.Q   = maxQ
        elif self.Q + dQ < minQ:
            self.Q   = minQ
        else:
            self.Q = self.Q + dQ
        self.nb_FR_mut += 1
        # self.calculate_breadth(testpanel, breadth_threshold,testpanelSize)
        # self.update_history()       

    def shm(self,cycle_number):
        """ Run somatic hypermutation and return self + new B cell clones. """
        
        # get number of cells that mutate, remove from the clone
        new_clones = []
        n_mut      = np.random.binomial(self.nb, p_mut)
        self.nb   -= n_mut
            
        # get number of CDR vs framework (FR) mutations
        n_CDR = np.random.binomial(n_mut, p_CDR)
        n_FR  = n_mut - n_CDR
            
        # process CDR mutations
        n_die, n_silent, n_affect  = np.random.multinomial(n_CDR, pvals = [p_CDR_lethal, p_CDR_silent, p_CDR_affect])
        self.nb                   += n_silent #add silent mutations to the parent clone
        for i in range(n_affect):
            b = BCell.clone(self)
            Ag = b.pick_Antigen() if b.nb_Ag > 1 else b.antigens[0]
            b.mutate_CDR(Ag)
            new_clones.append(b)
        
        # process FR mutations
        n_die, n_silent, n_affect  = np.random.multinomial(n_FR, pvals = [p_FR_lethal, p_FR_silent, p_FR_affect])
        self.nb                   += n_silent
        for i in range(n_affect):
            b = BCell.clone(self)
            b.mutate_FR()
            new_clones.append(b)

        # return the result
        if (self.nb>0): new_clones.append(self)
        return new_clones


###### Main functions ######
time_Steps = [2, 420, 780,900]

def sim_mixture(seed, sel_vec, antigen_list, testpanel, verbose=False):
    """ Simulate the affinity maturation process in a single germinal center (GC) and save the results to a CSV file. """
    
    # Run multiple trials and save all data to file
    np.random.seed(seed)
    # print(f'starting run {seed}')
    start    = timer()
    
    sel_antigens = [antigen_list[i] for i in range(len(antigen_list)) if sel_vec[i]==1]
    # print(sel_vec)
    # print(antigen_list)
    # print(sel_antigens)
    print(f'{len(sel_antigens)} antigens available')

    dicAgs = {c:sel_antigens for c in list(range(time_Steps[0],time_Steps[1]))}
    dicAgs.update({c:sel_antigens for c in list(range(time_Steps[1],time_Steps[2]))})
    concseq = 0.84/len(sel_antigens)
    dicconc = {c:concseq for c in list(range(time_Steps[0], time_Steps[1]))}
    dicconc.update({c:concseq for c in list(range(time_Steps[1], time_Steps[2]))})

    # fseed = open('output/seed.csv','w')
    # fendmut = open('output/output-endmut.csv', 'w')
    # fend  = open('output/output-end.csv', 'w')
    # ftot  = open('output/output-total.csv',  'w')
    # fbig  = open('output/output-largest-clone.csv', 'w')
    # fsurv = open('output/output-surv.csv','w')
    
    resstring = ",".join([f'res{i}' for i in range(length)])
    # fend.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,'+resstring+'\n')
    # fendmut.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,mut,'+resstring+'\n')
    # ftot.write('trial,cycle,number recycled,number exit,mean E,mean Ec,mean breadth,mean nb CDR mut\n')
    # fbig.write('trial,cycle,update,generation,CDR_mutations,E,delta_res,mut_res_index,'+resstring+'\n')
    # fsurv.write('trial,cycle,survival rate\n')
    # fseed.write('trial,exit_cycle,number,generation,CDR_mutations,E,Ec,breadth,mut,'+resstring+'\n')

    mean_breadth = np.zeros(nb_trial)
    mean_titer = np.zeros(nb_trial)
    # Events of a trial
    for t in range(nb_trial):
        
        gc.collect()
        # print_update(t, nb_trial)   # status check

        # INITIALIZATION - DEFINE DATA STRUCTURES
        recycled_cells   = []
        exit_cells       = [] # cells at the end of the simulation
        memory_cells     = [] # exit cells from previous cycles
        nb_recycled      = []
        nb_exit          = []
        memory_founders  = []
        GC_survival_rate = [] # nb of B cells after selection / nb of B cells after dark zone 

        # CYCLES 1 + 2 - CREATE FOUNDERS AND REPLICATE WITHOUT MUTATION
        nb_founders = nb_GC_founders #3   # number of founder B cells for a GC
        id_seeding_cells = np.random.choice(len(seedingCells), nb_founders, replace = False)
        # print(id_seeding_cells)        
        B_cells = [BCell(res = seedingCells[id_seeding_cells[i]]) for i in range(nb_founders)]
                
        # Update data
        #cycle 0
        nb_recycled.append(nb_founders)                     # all founders are recycled
        nb_exit.append(0)                                   # no founders exit the GC
        recycled_cells.append([deepcopy(b) for b in B_cells]) # add all cells of all 3 clones       
        
        #cycle 1
        nb_recycled.append(np.sum([b.nb for b in B_cells])) # all founders replicate and are recycled
        recycled_cells.append([deepcopy(b) for b in B_cells]) # add all cells of all 3 clones
        nb_exit.append(0)                                   # no founders exit
        
        # AFFINITY MATURATION
        GC_size_max  = nb_recycled[-1]  # maximum number of cells in the GC (= initial population size)
        
        cycle_number = 2
        nb_cycle_max = 199 if flag == 1 else (len(dicAgs) + cycle_number - 1) # maximum number of GC cycles
        # print("Nb Cycle Max : ", nb_cycle_max)
        t1 = timer()
        while (cycle_number < nb_cycle_max): 
             
            cycleAntigens = np.array(dicAgs[cycle_number])

            nb_Ag = find_nb_Ag(cycleAntigens)

            cycleconc = dicconc[cycle_number]
            cycledur  = dicGCDur[cycle_number]
            
            if cycle_number < cycledur: # keep same GC
                # print(f'runnung same GC cycle {cycle_number}')
                B_cells, out_cells, GC_surv_fraction = run_GC_cycle(B_cells, cycleAntigens, cycleconc, nb_Ag, cycle_number)
                # print(GC_surv_fraction)
                GC_survival_rate.append(GC_surv_fraction)
            elif cycle_number == cycledur: # start new GC
                # print('starting new GC at cycle number %d' % (cycle_number))
                memory_founders = pick_memCells_for_new_GC(memory_cells, nb_GC_founders) 
                # for b in memory_founders:
                #     writestring = '%d,'*5 + '%lf,'*(length+3)
                #     bres = tuple([b.res[i] for i in range(length)])
                #     inittuple = tuple([t, b.cycle_number, b.nb, b.generation, b.nb_CDR_mut, b.E, b.Ec, b.breadth])
                    
                #     # fseed.write(writestring % (inittuple+bres))
                #     # fseed.write('\n')
                B_cells, out_cells, GC_surv_fraction = run_GC_cycle(memory_founders, cycleAntigens, cycleconc, nb_Ag, cycle_number)
                GC_survival_rate.append(GC_surv_fraction)
            else:
                # continue 
                print('error in starting a GC')
                print(cycle_number)                
            
            # total number of cells in the GC
            GC_size = np.sum([b.nb for b in B_cells])
            
            # at the end, all B cells exit the GC
            if (cycle_number == nb_cycle_max-1) or (GC_size > GC_size_max):
                # if (cycle_number == nb_cycle_max-1):
                #     print("GC Ends, Time Limit, Cycle Number", cycle_number)
                # elif (GC_size>GC_size_max):
                #     print("GC Ends, Size Limit, Cycle Number", cycle_number)
                out_cells += B_cells
                nb_exit.append(np.sum([b.nb for b in out_cells]))
            else:
                memory_cells += out_cells
                nb_exit.append(np.sum([b.nb for b in out_cells]))
                out_cells = []
            
            recycled_cells.append([deepcopy(b) for b in B_cells])
            exit_cells.append(out_cells)
            nb_recycled.append(GC_size)
           
            if nb_recycled[-1] == 0:
                print(f'number of recycled cells {nb_recycled[-1]} at cycle no. {cycle_number}')
                if cycle_number < cycledur:
                    cycle_number = cycledur
                else:

                    break
            elif GC_size > GC_size_max:
                cycle_number = cycledur
            else:
                cycle_number += 1
            
            # fseed.flush()
        t2 = timer()
        # print(f'time in GC: {t2-t1}')

        
        i = -1    

        test_energies = np.array([b.energy(testpanel[j]) for j in range(testpanelSize) for b in recycled_cells[i]])
        titer_count = np.array([b.nb for j in range(testpanelSize) for b in recycled_cells[i]])
        mean_breadth[t] = np.sum(test_energies > breadth_threshold)/testpanelSize
        mean_titer[t] = np.sum((test_energies > breadth_threshold)*titer_count)/testpanelSize


    # print(f'mean_panel_breadth = {10000*np.mean(mean_breadth)}')
    # print(f'mean_titer = {np.mean(mean_titer)}')
    end = timer()
    # print(f'\nTotal time for {seed}: {(end - start)}, average per trial {(end - start)/float(nb_trial)}s')
    return mean_breadth, mean_titer

def find_nb_Ag(antigens):
    nb_Ag = 1 if np.shape(antigens)[0]==length else np.shape(antigens)[0]
    return nb_Ag
       
def print_update(current, end, bar_length=20):
    """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
    
    percent = float(current) / end
    dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
    space   = ''.join([' ' for k in range(bar_length - len(dash))])

    sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
    sys.stdout.flush()

def updating_antigens(B_cells, cycleAntigens):
    """ The antigens for all B cells are updated with the beginning of a new cycle. """
    for b in B_cells:
        b.update_antigens(cycleAntigens)    
    return B_cells
    
def run_dark_zone(B_cells, cycle_number, nb_rounds = 2):
    """ B cells proliferate and undergo SHM in the dark zone. """
    for i in range(nb_rounds):
        new_cells = []
        for b in B_cells:
            b.divide(cycle_number)
            new_cells += b.shm(cycle_number)
        B_cells = new_cells
    return B_cells

# def run_dark_zone(B_cells, cycle_number, nb_rounds = 2):
#     """ B cells proliferate and undergo SHM in the dark zone. """
    
#     for i in range(nb_rounds):
#         new_cells = []
#         temp = [(b.divide(cycle_number),b.shm(cycle_number)) for b in B_cells]
#         temp = [item for sublist in temp for item in sublist]
#         new_cells = list(filter(lambda item: item is not None, temp))

#         B_cells = [item for sublist in new_cells for item in sublist]

#     return B_cells


def run_binding_selection(B_cells, cycleconc, nb_Ag):
    """ Select B cells for binding to antigen. """
    
    new_cells=[]
    for b in B_cells:
        
        # one Ag at a time or all antigens at a time
        if all_Ag_per_cycle == True:
            Aglist = list(range(nb_Ag))
            b.last_bound = np.random.multinomial(b.nb, pvals = [1./float(nb_Ag)] * nb_Ag)
        else:
            random_Ag_i = np.random.randint(0,nb_Ag)
            Aglist = [random_Ag_i]
            b.picked_Ag = random_Ag_i
            b.last_bound = [0]*random_Ag_i + [b.nb] + [0]*(nb_Ag-random_Ag_i-1)
            
        # compute binding energy and chance of death ( = 1 - chance of survival )
        factor = 0
        for i in Aglist:
            Ag_bound      = np.exp(energy_scale * (b.energy(b.antigens[i])-activation_energy))
            factor       += cycleconc * Ag_bound
        langmuir_conj = 1. / (1. + factor)

        # remove dead cells and update binding details
        for i in Aglist:
            n_die            = np.random.binomial(b.last_bound[i], langmuir_conj)
            b.nb            -= n_die
            b.last_bound[i] -= n_die

        if b.nb>0:
            new_cells.append(b)
            
    return new_cells

def run_help_selection(B_cells, nb_Ag):
    """ Select B cells to receive T cell help. """
        
    # All antigens at a time
    if all_Ag_per_cycle:
        # get binding energies
        binding_energy     = [[b.energy(b.antigens[i]) for i in range(nb_Ag)] for b in B_cells]
        binding_energy_tot = []
        for i in range(len(B_cells)):
            for j in range(nb_Ag): binding_energy_tot += [binding_energy[i][j]] * B_cells[i].last_bound[j]

        # cells in the top (help_cutoff) fraction of binders survive
        if len(binding_energy_tot) > 0:
            cut_idx       = np.max([0, int(np.floor(help_cutoff * len(binding_energy_tot)))-1])
            energy_cutoff = np.array(binding_energy_tot)[np.argsort(binding_energy_tot)][::-1][cut_idx]
            n_die_tie     = len(binding_energy_tot) - cut_idx - np.sum(binding_energy_tot < energy_cutoff)

            # kill all B cells below threshold
            for i in np.random.permutation(len(B_cells)):
                for j in np.random.permutation(nb_Ag):
                    energy = binding_energy[i][j]
                    if energy < energy_cutoff:
                        B_cells[i].nb            -= B_cells[i].last_bound[j]
                        B_cells[i].last_bound[j]  = 0
                    elif (energy == energy_cutoff) and (n_die_tie > 0):
                        if B_cells[i].last_bound[j] < n_die_tie:
                            B_cells[i].nb            -= B_cells[i].last_bound[j]
                            n_die_tie                -= B_cells[i].last_bound[j]
                            B_cells[i].last_bound[j]  = 0
                        else:
                            B_cells[i].nb            -= n_die_tie
                            B_cells[i].last_bound[j] -= n_die_tie
                            n_die_tie                 = 0
        cells_surv = np.sum([b.nb for b in B_cells]) 
        
    # one Ag at a time 
    else:
        # get binding energies
        binding_energy     = [[b.energy(b.antigens[i]) for i in [b.picked_Ag]] for b in B_cells]
        binding_energy_tot = []
        for i,b in enumerate(B_cells):
            for j in [0]: binding_energy_tot += [binding_energy[i][j]] * B_cells[i].last_bound[j]

        # cells in the top (help_cutoff) fraction of binders survive
        if len(binding_energy_tot) > 0:
            cut_idx       = np.max([0, int(np.floor(help_cutoff * len(binding_energy_tot)))-1])
            energy_cutoff = np.array(binding_energy_tot)[np.argsort(binding_energy_tot)][::-1][cut_idx]
            n_die_tie     = len(binding_energy_tot) - cut_idx - np.sum(binding_energy_tot < energy_cutoff)

            # kill all B cells below threshold
            for i,b in enumerate(np.random.permutation(B_cells)):
                for j in np.random.permutation([0]):
                    energy = binding_energy[i][j]
                    if energy < energy_cutoff:
                        B_cells[i].nb            -= B_cells[i].last_bound[j]
                        B_cells[i].last_bound[j]  = 0
                    elif (energy == energy_cutoff) and (n_die_tie > 0):
                        if B_cells[i].last_bound[j] < n_die_tie:
                            B_cells[i].nb            -= B_cells[i].last_bound[j]
                            n_die_tie                -= B_cells[i].last_bound[j]
                            B_cells[i].last_bound[j]  = 0
                        else:
                            B_cells[i].nb            -= n_die_tie
                            B_cells[i].last_bound[j] -= n_die_tie
                            n_die_tie                 = 0
        cells_surv = np.sum([b.nb for b in B_cells])   
        
    return B_cells, cells_surv
    

def run_recycle(B_cells):
    """ Randomly select B cells to be recycled back into the GC or to exit. """

    new_cells  = []                                 # cells that will remain in the GC
    exit_cells = []                                 # cells that will exit the GC
    n_tot      = np.sum([b.nb for b in B_cells])    # total number of cells currently in the GC
    n_exit     = int(np.floor(p_exit * n_tot))      # number of cells that will exit the GC
    b_exit     = np.array([])                       # index of cells that will exit the GC

    if (n_tot > 0) and (n_exit > 0):
        b_exit = np.random.choice(n_tot, n_exit, replace=False)

    idx = 0
    for b in B_cells:
    
        # find which cells exit the GC
        n_exit  = np.sum((idx <= b_exit) * (b_exit < idx + b.nb))
        idx    += b.nb
        b.nb   -= n_exit
        
        # add remainder to recycled cells
        if (b.nb > 0):
            new_cells.append(b)
    
        # record exit cells
        if (n_exit > 0):
            exit_cells.append(deepcopy(b))
            exit_cells[-1].nb = n_exit

    return new_cells, exit_cells

def pick_memCells_for_new_GC(memory_cells, nb_GC_founders):
    num_mem_seed = nb_GC_founders - num_naive_seed
    n_mem_cells = len(memory_cells)
    id_new_founders = np.random.choice(n_mem_cells, num_mem_seed, replace=False)
    mem_founders = [memory_cells[id_new_founders[i]] for i in range(num_mem_seed)]
    
    id_seeding_cells = np.random.choice(len(seedingCells), num_naive_seed, replace = False)
    naive_founders = [BCell(res = seedingCells[id_seeding_cells[i]]) for i in range(num_naive_seed)]
    new_founders = mem_founders + naive_founders
    return new_founders

def run_breadth_calculation(panel_energies, threshold, panelSize):
    average  = np.mean(panel_energies)
    variance = np.var(panel_energies)
    breadth  = float(sum(x > threshold for x in panel_energies))/panelSize 
    return average, variance, breadth

def run_GC_cycle(B_cells, cycleAntigens, cycleconc, nb_Ag, cycle_number):
    """ Run one cycle of the GC reaction. """
    B_cells = updating_antigens(B_cells, cycleAntigens)         # UPDATE antigens
    B_cells = run_dark_zone(B_cells, cycle_number)              # DARK  ZONE - two rounds of division + SHM + updates cycle_number
    total_cells = np.sum([b.nb for b in B_cells])
    
    if total_cells == 0: 
        print('GC extinct at cycle ', cycle_number)
        exit_cells = [] 
        GC_surv_fraction = 0
    else: 
        B_cells = run_binding_selection(B_cells, cycleconc, nb_Ag)  # LIGHT ZONE - selection for binding to Ag
        B_cells, cells_surv = run_help_selection(B_cells, nb_Ag)    # LIGHT ZONE - selection to receive T cell help
        GC_surv_fraction = float(float(cells_surv)/float(total_cells))
        B_cells, exit_cells = run_recycle(B_cells)
    
    return B_cells, exit_cells, GC_surv_fraction               # RECYCLE - randomly pick exiting cells from the surviving B cells




