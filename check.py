from issd import issd
from miniutils import *
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from figutils import *

def check(X, y):
    """
    X: input data (n_samples, n_features)
    y: label (n_samples, )
    """
    dim_2 = X.shape[1]
    if dim_2 !=1:
        X_pca = PCA(n_components=1).fit_transform(X)
        # X_pca = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)
    print(X_pca.shape)
    true_matrix, cut_points = calculate_true_matrix(state_seq)
    segments = [X[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    # print(len(segments), len(cut_points), cut_points, X.shape)
    matrix = pair_wise_nntest(segments, 30, 50)
    # scale to 0,1
    # matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    print(matrix.shape)

    ind_inner = np.where(true_matrix == 1)
    ind_inter = np.where(true_matrix == 0)
    max_inner = np.max(matrix[ind_inner])
    min_inter = np.min(matrix[ind_inter])
    print(max_inner, min_inter, max_inner<=min_inter)

    return matrix, true_matrix

# fname_list = os.listdir('data/MoCap/raw')
# fname_list = os.listdir('data/USC-HAD/raw')
# fname_list.sort()
# print(fname_list)
# for fname in fname_list:
#     data, state_seq = load_data(os.path.join('data/USC-HAD/raw', fname))
#     # data = data[:, [48, 38, 55, 51]]
#     data = data[:, [0,1,2,3]]
#     check(data, state_seq)

example_data, state_seq = load_data('data/MoCap/raw/86_01.npy')
example_data = example_data[:, [48, 38, 55, 51]]
matrix, true_matrix = check(example_data, state_seq)

heatmap_with_dots(matrix, true_matrix)
plt.savefig('result.png')
plt.close()
bartchart_with_class(matrix, true_matrix)
plt.savefig('bar.png')


# import matplotlib.pyplot as plt
# plt.figure(figsize=(5, 5))
# plt.imshow(matrix, cmap='YlGnBu', interpolation='nearest')
# plt.colorbar()
# # adjust the colorbar, adapt to the size of the figure
# plt.subplots_adjust(right=0.8)
# # draw a red box for the grid of inner class
# ind_inner = np.where(true_matrix == 1)
# for i in range(len(ind_inner[0])):
#     plt.plot(ind_inner[1][i], ind_inner[0][i], 'ro')
# # draw a blue box for the grid of inter class
# ind_inter = np.where(true_matrix == 0)
# for i in range(len(ind_inter[0])):
#     plt.plot(ind_inter[1][i], ind_inter[0][i], 'go')
# plt.tight_layout()
# plt.savefig('result.png')

# import matplotlib.pyplot as plt
# # show the two matrix in the same figure
# plt.figure(figsize=(10, 5))
# # plt.subplot(121)
# plt.imshow(matrix, cmap='YlGnBu', interpolation='nearest')
# # plt.subplot(122)
# # plt.imshow(true_matrix, cmap='gray', interpolation='nearest')
# # plt.colorbar()
# # set a veritical title for the colorbar
# plt.subplots_adjust(right=0.8)
# cbar_ax = plt.gcf().axes[-1]
# cbar_ax.set_ylabel('Difference')
# # adjust the colorbar, adapt to the size of the figure
# plt.tight_layout()
# plt.savefig('result.png')