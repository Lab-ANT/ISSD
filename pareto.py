import os
import numpy as np
from miniutils import *
import itertools

dataset = 'MoCap'
n_components = 2
fname_list = os.listdir(f'data/{dataset}/raw')
os.makedirs(f'data/issd', exist_ok=True)

print(fname_list)

other_solutions = []

def objective_interval(solution, mean_interval):
    return np.mean(mean_interval[solution])

def objective_completeness(solution, indicator_matrices):
    return np.sum(matrix_OR(indicator_matrices[solution]))

def evaluate_objectives(solution, mean_interval, indicator_matrices):
    other_solutions.append([objective_interval(solution, mean_interval), objective_completeness(solution, indicator_matrices)])
    return [objective_interval(solution, mean_interval), objective_completeness(solution, indicator_matrices)]

def initialize_solution():
    return [1]

def dominates(objectives1, objectives2):
    for obj1, obj2 in zip(objectives1, objectives2):
        if obj1 < obj2:
            return False
    return True

def update_pareto_front(pareto_front, solution, new_objectives):
    to_be_removed = []

    for i, existing_objectives in enumerate(pareto_front):
        if dominates(new_objectives, existing_objectives):
            to_be_removed.append(i)
        elif dominates(existing_objectives, new_objectives):
            return pareto_front, solution

    for index in reversed(to_be_removed):
        pareto_front.pop(index)
        solution.pop(index)

    pareto_front.append(new_objectives)
    solution.append(new_solution)

    return pareto_front, solution

for fname in fname_list:
    channel_metirces = []
    channel_interval = []
    channel_max_inner = []
    for fn_test in fname_list:
        if fn_test == fname:
            continue
        matrices = np.load(f'output/issd-qf/{dataset}/{fn_test[:-4]}/matrices.npy')
        max_inner = np.load(f'output/issd-qf/{dataset}/{fn_test[:-4]}/max_inner.npy')
        mean_inter = np.load(f'output/issd-qf/{dataset}/{fn_test[:-4]}/mean_inter.npy')
        mean_inner = np.load(f'output/issd-qf/{dataset}/{fn_test[:-4]}/mean_inner.npy')
        interval = mean_inter - mean_inner
        channel_interval.append(interval)
        # indicator_matrices = [m>tau for m, tau in zip(matrices, max_inner)]
        channel_metirces.append(matrices)
        channel_max_inner.append(max_inner)
    channel_max_inner = np.array(channel_max_inner).mean(axis=0)
    channel_interval = np.array(channel_interval).mean(axis=0)
    # print(channel_interval, channel_max_inner)
    per_channel_list = []
    num_channels = channel_interval.shape[0]
    num_ts = len(fname_list)-1
    for i in range(num_channels):
        list1 = []
        for j in range(num_ts):
            list1.append(channel_metirces[j][i].copy().flatten())
        per_channel_list.append(np.concatenate(list1))
    per_channel_list = np.array(per_channel_list)
    # print(per_channel_list)

    masked_idx = np.array([False]*len(per_channel_list))
    indicator_matrices = [m>tau for m, tau in zip(per_channel_list, channel_max_inner)]
    indicator_matrices = np.array(indicator_matrices)

combinations = list(itertools.combinations(range(num_channels), n_components))
solutions = iter(combinations)
current_solution = list(next(solutions))
solution = [current_solution]
pareto_front = [evaluate_objectives(current_solution, channel_interval, indicator_matrices)]
# pareto_front = [evaluate_objectives(current_solution, channel_interval, per_channel_list)]
max_iterations = int(len(combinations)-1)
for iteration in range(max_iterations):
    new_solution = list(current_solution)
    new_objectives = evaluate_objectives(new_solution, channel_interval, indicator_matrices)
    # new_objectives = evaluate_objectives(new_solution, channel_interval, per_channel_list)
    pareto_front, solution = update_pareto_front(pareto_front, solution, new_objectives)
    current_solution = next(solutions)
print(pareto_front, solution)
pareto_optimal = np.array(pareto_front)
# solution_idx = np.argmin(pareto_optimal[:,0])
solution_idx = np.argmax(pareto_optimal[:,0])
selected_channels = solution[solution_idx]
data, state_seq = load_data(os.path.join(f'data/{dataset}/raw/', fname))
data_reduced = data[:,selected_channels]
data_reduced = np.vstack((data_reduced.T, state_seq)).T
np.save(os.path.join(f'data/{dataset}/issd', fname), data_reduced)

import matplotlib.pyplot as plt
pareto_optimal = np.array(pareto_front)
pareto_optimal = pareto_optimal[np.argsort(pareto_optimal[:, 0])]
other_solutions = np.array(other_solutions)
plt.plot(pareto_optimal[:, 0], pareto_optimal[:, 1], c='r')
plt.scatter(other_solutions[:, 0], other_solutions[:, 1], label='4', s=0.5)
plt.scatter(pareto_optimal[:, 0], pareto_optimal[:, 1], label='4', c='r')
plt.savefig('pareto.png')
plt.close()

non_trival_idx = exclude_trival_segments(state_seq, 50)
state_seq = state_seq[non_trival_idx]
data = data[non_trival_idx]
solution = np.array(solution)
solution = solution[np.argsort(pareto_optimal[:, 0])]
matrix_solution = []
for s in solution:
    true_matrix, cut_points = calculate_true_matrix(state_seq)
    segments = [data[cut_points[i]:cut_points[i+1],[s]] for i in range(len(cut_points)-1)]
    matrix = pair_wise_nntest(segments, 50, 20, method='nn')
    matrix_solution.append(matrix)
num_solutions = len(solution)
width = int(math.sqrt(num_solutions))+1
fig, ax = plt.subplots(nrows=width, ncols=width, figsize=(20,20))
for i in range(width):
    for j in range(width):
        ax[i,j].set_yticks([])
        ax[i,j].set_xticks([])
        if i*width+j >= num_solutions:
            continue
        else:
            ax[i,j].imshow(matrix_solution[i*width+j], cmap='gray')
plt.savefig( 'matrices.png')
plt.close()