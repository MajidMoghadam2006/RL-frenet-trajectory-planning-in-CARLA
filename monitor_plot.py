import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

'''
Utility Folder in order to visualize Rewards
Usage: place agent folder in logs/agent_xx
args: 
--agent_ids: agents to be plotted (default = None)
--window_size: window size of moving (average default = 100)
--legends: : ['DDPG','TRPO']
--colors: list: ['red','blue']
--alpha: opacity
example usage:
python reward_plotter.py --agent_ids 26 27 28 --window_size=10
'''


def plot_rewards(agents, window_size=100, colors=None, alpha=0.2, lr=None, n_timesteps=float('inf')):
    data = []

    for i in agents:
        data.append(pd.read_csv('logs/agent_{}/monitor.csv'.format(i), skiprows=1))

    average = []
    std_dev = []
    step_cum = []

    for x in range(len(data)):
        temp = []
        temp_step = []
        temp_std = []
        sum_ = 0

        for i in range(window_size - 1):
            # temp.append(np.mean(data[x]['r'][:i]))
            temp.append(0)
            sum_ += data[x]['l'][i]
            temp_step.append(sum_)

        for i in range(window_size - 1, data[x]['r'].shape[0]):
            temp.append(np.mean(data[x]['r'][i - window_size - 1:i]))
            temp_std.append(np.std(data[x]['r'][i - window_size - 1:i]))
            sum_ += data[x]['l'][i]
            temp_step.append(sum_)
            if sum_ > n_timesteps:
                break

        average.append(temp)
        std_dev.append(temp_std)
        step_cum.append(temp_step)

    plt.figure(figsize=(12, 8))

    if lr is None:
        lr = agents
    if colors is None:
        colors = [np.random.rand(3, ) for x in agents]

    for i in range(len(lr)):
        plt.plot(step_cum[i], average[i], '-', color=colors[i])
        plt.fill_between(step_cum[i][window_size - 1:], np.subtract(average[i][window_size - 1:], std_dev[i]),
                         np.add(average[i][window_size - 1:], std_dev[i]), color=colors[i], alpha=alpha)

    plt.title('CARLA')
    plt.xlabel('TimeSteps')
    plt.ylabel('Mean_Reward-{}'.format(window_size))
    plt.legend(lr)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ids', nargs='+', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--colors', nargs='+', type=str, default=None)
    parser.add_argument('--lr', nargs='+', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--n_steps', type=float, default=1e7)
    args = parser.parse_args()
    plot_rewards(args.agent_ids, args.window_size, args.colors, args.alpha, args.lr, args.n_steps)
