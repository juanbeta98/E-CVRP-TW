from E_CVRP_TW import E_CVRP_TW, Experiment
from multiprocess import pool
import os
import pickle

path = f'{os.getcwd()}/'

if __name__ == '__main__':
    env = E_CVRP_TW(path)
    lab:Experiment = Experiment(path, False, True)

    test_bed = env.instances[:int(len(env.instances)/2)]

    p = pool.Pool(processes = 6)
    p.map(lab.experimentation, test_bed)
    p.terminate()


