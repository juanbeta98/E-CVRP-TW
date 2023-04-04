import pickle
import matplotlib.pyplot as plt
import sys

def retrieve_const_performance(instance, path):

    file = open(path + f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()

    return data


def plot_const_performance(instance, path):

    data = retrieve_const_performance(instance, path)

    plt.plot(data['inc times'], data['incumbents'], color = 'purple')
    plt.title(f"Constructive's performance {instance}")
    plt.xlabel('Time (s)')
    plt.ylabel('Incumbent')

    plt.show()
