import sys
import pickle
import pandas as pd
path: str = '/Users/juanbeta/My Drive/Research/Energy/E-CVRP-TW/Code/'
#path: str = 'C:/Users/jm.betancourt/Documents/Research/Energy//E-CVRP-TW/Code/'

from E_CVRP_TW import  E_CVRP_TW, Feasibility
env = E_CVRP_TW(path)

sys.path.insert(0,path+'Experimentation/')
import plot_performance as plot


instance = 'c101_21.txt'
env.load_data(instance)
env.generate_parameters()

feas_op = Feasibility()

individual = [['D', 'C23', 'C25', 'C30', 'C28', 'C26', 'C10', 'C9', 'C8', 'D'],
 ['D',
  'C40',
  'C64',
  'C44',
  'C45',
  'C48',
  'C46',
  'C47',
  'C43',
  'C50',
  'C49',
  'S13',
  'D'],
 ['D', 'C21', 'C24', 'C29', 'C32', 'C33', 'C31', 'C51', 'S13', 'D'],
 ['D', 'C22', 'C34', 'C36', 'C39', 'C38', 'C37', 'C35', 'C52', 'S13', 'D'],
 ['D', 'C69', 'C72', 'C61', 'C66', 'D'],
 ['D', 'C65', 'C62', 'C68', 'C54', 'C55', 'C57', 'S15', 'D'],
 ['D', 'C7', 'C5', 'C91', 'D'],
 ['D', 'C67', 'C74', 'C42', 'C56', 'C53', 'S16', 'C88', 'C89', 'D'],
 ['D', 'C75', 'C2', 'C3', 'C13', 'D'],
 ['D', 'C63', 'C86', 'C82', 'C84', 'C85', 'C99', 'S3', 'D'],
 ['D', 'C27', 'C4', 'C6', 'C11', 'D'],
 ['D', 'C20', 'C1', 'C98', 'C95', 'C94', 'C92', 'C93', 'C97', 'S3', 'D'],
 ['D', 'C41', 'C60', 'C58', 'S16', 'C80', 'C81', 'S19', 'D'],
 ['D', 'C90', 'C87', 'C83', 'C78', 'S19', 'D'],
 ['D', 'C96', 'C100', 'S3', 'D'],
 ['D', 'C17', 'C15', 'D'],
 ['D', 'C12', 'C14', 'C16', 'C19', 'C18', 'S9', 'D'],
 ['D', 'C59', 'S14', 'C79', 'C77', 'C76', 'S20', 'D'],
 ['D', 'C73', 'C70', 'C71', 'S20', 'D']]

visited = list(); count = 0

feasible, _ = feas_op.individual_check(env, individual, complete=True)
print(feasible)
for route in individual:
    for pos in route:
        if env.node_type[pos] == 'c' and pos not in visited:
            visited.append(pos)
            count += 1
print(f'The individual covers {count} costumers')
