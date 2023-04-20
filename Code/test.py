import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW()

sys.path.insert(0,path+'Experimentation/')
import plot_performance as plot

operators = ['Darwinian phi rate', 'two opt']
data_op = dict()

test_bed = ['c202C10.txt']#, 'c103_21.txt']
instance = test_bed[0]

for instance in test_bed:
    data_op[instance] = dict()
    for operator in operators[1:]:
        data_op[instance][operator] = plot.retrieve_const_performance(instance, path+f'Experimentation/Operators/{operator}/')

env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()
feasible, _ = feas_op.individual_check(env, data_op[test_bed[0]]['two opt']['best individual'])
print(feasible)