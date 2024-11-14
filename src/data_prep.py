import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import expm_multiply
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.animation as animation

def get_laplacian_US_ROADS(graph_file_path, coord_file_path,
                           x_min=-125, x_max=-70, y_min=24, y_max=49,
                           graph_skip_lines=15, coord_skip_lines=7):
    """
    Loads US roads graph and coordinates from given file paths, filters nodes based on
    geographical boundaries, and returns the Laplacian matrix along with filtered coordinates and mapped edge indices.

    Parameters:
    - graph_file_path (str): Path to the graph file in Matrix Market (.mtx) format.
    - coord_file_path (str): Path to the coordinates file in Matrix Market (.mtx) format.
    - x_min, x_max (float): Minimum and maximum longitude for filtering nodes.
    - y_min, y_max (float): Minimum and maximum latitude for filtering nodes.
    - graph_skip_lines (int): Number of header lines to skip in the graph file.
    - coord_skip_lines (int): Number of header lines to skip in the coordinates file.

    Returns:
    - laplacian_matrix (scipy.sparse.csr_matrix): The Laplacian matrix of the filtered graph.
    - filtered_x (np.ndarray): X-coordinates of the filtered nodes.
    - filtered_y (np.ndarray): Y-coordinates of the filtered nodes.
    - mapped_ind_i (list): List of source node indices for edges.
    - mapped_ind_j (list): List of target node indices for edges.
    """
    # Load edges from the graph file
    try:
        with open(graph_file_path, 'r') as file:
            for _ in range(graph_skip_lines):
                next(file)
            edges = np.loadtxt(file, dtype=int)
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file not found at path: {graph_file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the graph file: {e}")

    # Adjust indices to be zero-based
    ind_i = edges[:, 0] - 1
    ind_j = edges[:, 1] - 1

    # Load coordinates from the coordinates file
    try:
        with open(coord_file_path, 'r') as file:
            for _ in range(coord_skip_lines):
                next(file)
            data = np.loadtxt(file)
        
        if data.size % 2 != 0:
            raise ValueError("Coordinate data does not contain pairs of x and y values.")
        
        num_vertices = data.size // 2
        x = data[:num_vertices]
        y = data[num_vertices:]
    except FileNotFoundError:
        raise FileNotFoundError(f"Coordinate file not found at path: {coord_file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the coordinate file: {e}")

    # Define geographical boundaries
    valid_nodes_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    valid_node_indices = np.where(valid_nodes_mask)[0]

    if valid_node_indices.size == 0:
        raise ValueError("No nodes found within the specified geographical boundaries.")

    # Filter coordinates
    filtered_x = x[valid_nodes_mask]
    filtered_y = y[valid_nodes_mask]

    # Map old node indices to new indices after filtering
    node_map = {old_index: new_index for new_index, old_index in enumerate(valid_node_indices)}

    # Filter edges to include only those between valid nodes
    valid_edges_mask = np.isin(ind_i, valid_node_indices) & np.isin(ind_j, valid_node_indices)
    filtered_ind_i = ind_i[valid_edges_mask]
    filtered_ind_j = ind_j[valid_edges_mask]

    if filtered_ind_i.size == 0:
        raise ValueError("No edges found between nodes within the specified geographical boundaries.")

    # Map old indices to new indices
    mapped_ind_i = [node_map[i] for i in filtered_ind_i]
    mapped_ind_j = [node_map[j] for j in filtered_ind_j]

    n_filtered_nodes = len(filtered_x)

    # Combine both directions for each edge to ensure symmetry
    all_mapped_ind_i = np.concatenate([mapped_ind_i, mapped_ind_j])
    all_mapped_ind_j = np.concatenate([mapped_ind_j, mapped_ind_i])

    # Create the symmetric adjacency matrix
    adjacency_matrix = csr_matrix(
        (np.ones(len(all_mapped_ind_i)), (all_mapped_ind_i, all_mapped_ind_j)),
        shape=(n_filtered_nodes, n_filtered_nodes)
    )

    # Compute the Laplacian matrix
    laplacian_matrix = laplacian(adjacency_matrix, normed=False)

    print(f"Laplacian is a sparse matrix with format: {laplacian_matrix.format}")

    return laplacian_matrix, filtered_x, filtered_y, mapped_ind_i, mapped_ind_j