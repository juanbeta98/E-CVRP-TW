import pickle
import matplotlib.pyplot as plt

def plot_constructive(instance):
    file = open(f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()

    plt.plot(list(data['incumbents'], data['inc times']))
