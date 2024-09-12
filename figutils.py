import matplotlib.pyplot as plt
import numpy as np

def heatmap_with_dots(value_matrix, bool_matrix,
                           dot_size=None,
                           cmap='YlGnBu'):
    # Check if both matrices have the same shape
    assert value_matrix.shape == bool_matrix.shape, "Matrices must have the same shape"
    
    plt.style.use('classic')
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(value_matrix, cmap=cmap, interpolation='nearest')
    # Show the plot
    plt.colorbar()
    
    # Get the indices of 0's and 1's in the boolean matrix
    zeros_indices = np.argwhere(bool_matrix == 0)
    ones_indices = np.argwhere(bool_matrix == 1)
    
    if dot_size is None:
        # adapt dot size to the size of the matrix
        dot_size = 1000 / len(value_matrix)

    # Draw green dots for 0's
    for (i, j) in zeros_indices:
        plt.scatter(j, i, s=dot_size, color='#5D8AA8')
    
    # Draw blue dots for 1's
    for (i, j) in ones_indices:
        # plt.scatter(j, i, s=dot_size, color='#A5CC82')
        plt.scatter(j, i, s=dot_size, color='green')

    # set xlim and ylim
    plt.xlim(-0.5, value_matrix.shape[1]-0.5)
    plt.ylim(value_matrix.shape[0]-0.5, -0.5)

    # adjust x and y label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

def bartchart_with_class(matrix, bool_matrix):
    """
    This function plots a bar chart from a numerical matrix with the arrangement based on a boolean matrix.
    
    Args:
    matrix (np.ndarray): A 2D array of numerical values.
    bool_matrix (np.ndarray): A 2D boolean matrix of the same shape as the numerical matrix.
    
    The function will plot values from positions with 'False' in the boolean matrix in blue, and
    positions with 'True' in green, arranging 'False' values before 'True' values.
    """
    # Flatten both matrices to make processing easier
    flat_matrix = matrix.flatten()
    flat_bool_matrix = bool_matrix.flatten()

    plt.style.use('classic')
    # Separate values based on the boolean matrix
    zero_values = flat_matrix[flat_bool_matrix == 0]
    one_values = flat_matrix[flat_bool_matrix == 1]

    # Create indices for plotting to maintain the order of 0s first, then 1s
    x_indices_zero = np.arange(len(zero_values))
    x_indices_one = np.arange(len(zero_values), len(zero_values) + len(one_values))

    # Plot the values in two different colors
    plt.figure(figsize=(8, 6))
    plt.bar(x_indices_zero, zero_values, color='#5D8AA8', label='Inner-state differences',width=0.5)
    plt.bar(x_indices_one, one_values, color='#A5CC82', label='Inter-state differences', width=0.5)

    # set xlim and ylim
    plt.xlim(-0.5, len(flat_matrix)-0.5)
    plt.ylim(np.min(flat_matrix), np.max(flat_matrix))
    
    # Add labels and title for clarity
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Chart Based on Boolean Matrix')
    plt.legend()
    plt.tight_layout()