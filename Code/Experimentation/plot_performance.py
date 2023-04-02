import pickle
import matplotlib.pyplot as plt
import sys

def plot_const_performance(instance, path):

    path += 'Constructive/'

    file = open(path + f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()

    plt.plot(data['inc times'], data['incumbents'], color = 'purple')
    plt.title(f"Constructive's performance {instance}")
    plt.xlabel('Time (s)')
    plt.ylabel('Incumbent')

    plt.show()
