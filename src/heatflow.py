
import numpy as np 
import time 
from scipy.sparse.linalg import expm_multiply

def compute_heat_evolution(laplacian_matrix, filtered_x, filtered_y, n_filtered_nodes,
                          diffusion_rate=0.5, center_point=(-118, 34),
                          radius=0.5, initial_heat=1.0,
                          time_end=140, num_steps=25):
    """
    Computes the heat distributions over time using the Laplacian matrix.

    Parameters:
    - laplacian_matrix (scipy.sparse.csr_matrix): The Laplacian matrix of the graph.
    - filtered_x (np.ndarray): X-coordinates of the nodes.
    - filtered_y (np.ndarray): Y-coordinates of the nodes.
    - n_filtered_nodes (int): Number of nodes after filtering.
    - diffusion_rate (float): Scaling factor for heat spread.
    - center_point (tuple): (x, y) coordinates for the initial heat source center.
    - radius (float): Radius around the center point to initialize heat.
    - initial_heat (float): Initial heat value to set within the radius.
    - time_end (float): The maximum time value for diffusion.
    - num_steps (int): Number of time steps to compute.

    Returns:
    - heat_distributions (list of np.ndarray): Heat distribution at each time step.
    - delta_heats (list of np.ndarray): Incremental heat distribution between time steps.
    - time_steps (np.ndarray): Array of time steps.
    """
    scaled_laplacian = diffusion_rate * laplacian_matrix
    u0 = np.zeros(n_filtered_nodes)
    x_center, y_center = center_point

    distances = np.sqrt((filtered_x - x_center)**2 + (filtered_y - y_center)**2)
    within_radius_indices = np.where(distances <= radius)[0]
    u0[within_radius_indices] = initial_heat

    # Define time steps for heat diffusion
    time_steps = np.linspace(0, time_end, num=num_steps) 

    # For logarithmic spacing, uncomment the following lines:
    # start_time_log = np.log10(0.1)   # Log of the minimum time
    # end_time_log = np.log10(time_end)
    # time_steps = np.logspace(start_time_log, end_time_log, num=num_steps)

    # Compute heat distributions using expm_multiply
    start = time.time()
    heat_distributions = [
        expm_multiply(-t * scaled_laplacian, u0) for t in time_steps #<------------- main time evolution
    ]
    time_expm_multiply = time.time() - start
    print(f"Time using expm_multiply: {time_expm_multiply:.4f} seconds")

    # Compute incremental heat distributions
    delta_heats = []
    for i in range(len(heat_distributions)):
        if i == 0:
            delta_heat = heat_distributions[0]
        else:
            delta_heat = heat_distributions[i] - heat_distributions[i - 1]
        # Set negative values to zero (only show new heat)
        delta_heat[delta_heat < 0] = 0
        delta_heats.append(delta_heat)

    return heat_distributions, delta_heats, time_steps