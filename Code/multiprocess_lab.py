from E_CVRP_TW import E_CVRP_TW, Experiment
from multiprocess import pool
import os
import pickle

path = f'{os.getcwd()}/'

if __name__ == '__main__':
    env = E_CVRP_TW(path)
    lab:Experiment = Experiment(path, False, True, False)

    test_bed = [env.sizes['s'][0]]#,env.sizes['m'][0],env.sizes['l'][0]]

    p = pool.Pool(processes = 4)
    results = p.map(lab.experimentation, test_bed)
    # print(results.get(timeout = 1))
    p.terminate()


