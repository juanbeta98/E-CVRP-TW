from E_CVRP_TW import E_CVRP_TW, Experiment
from multiprocess import pool
import os

path = f'{os.getcwd()}/'
if path[7:15] == 'juanbeta': computer = 'mac'
else: computer = 'pc'

if __name__ == '__main__':
    env = E_CVRP_TW(path)
    lab:Experiment = Experiment(path, False, True)

    test_bed = env.instances

    if computer == 'mac':   p = pool.Pool(processes = 8)
    else: p = pool.Pool()

    p.map(lab.experimentation, test_bed)
    p.terminate()