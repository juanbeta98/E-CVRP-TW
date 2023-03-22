import pickle
import matplotlib.pyplot as plt
import sys

def plot_const_performance(instance, path, station_placement):

    if station_placement == 'det': path += 'Constructive/Deterministic RCL Heu/'
    else: path += 'Constructive/Stochastic RCL Heu/'

    file = open(path + f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()

    plt.plot(data['inc times'], data['incumbents'], color = 'purple')
    plt.title(f"Constructive's ({station_placement}) performance {instance}")
    plt.xlabel('Time (s)')
    plt.ylabel('Incumbent')

    plt.show()

