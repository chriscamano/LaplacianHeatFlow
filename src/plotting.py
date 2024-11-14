import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy.ma as ma  
import imageio
import os  

def animate_heat(filtered_x, filtered_y, mapped_ind_i, mapped_ind_j,
                        delta_heats, time_steps, delta_heat_min=None, delta_heat_max=None,
                        output_dir='delta_heat_frames', gif_name='incremental_heat_diffusionCT.gif',
                        cmap='coolwarm', dpi=500, frame_duration=0.5):
    """
    Plots the incremental heat distributions and creates a GIF animation.

    Parameters:
    - filtered_x (np.ndarray): X-coordinates of the nodes.
    - filtered_y (np.ndarray): Y-coordinates of the nodes.
    - mapped_ind_i (list or np.ndarray): Mapped source node indices for edges.
    - mapped_ind_j (list or np.ndarray): Mapped target node indices for edges.
    - delta_heats (list of np.ndarray): Incremental heat distribution between time steps.
    - time_steps (np.ndarray): Array of time steps.
    - delta_heat_min (float, optional): Minimum delta heat value for normalization.
    - delta_heat_max (float, optional): Maximum delta heat value for normalization.
    - output_dir (str): Directory to save frame images.
    - gif_name (str): Name of the output GIF file.
    - cmap (str): Colormap for plotting.
    - dpi (int): Resolution of the saved frames.
    - frame_duration (float): Duration of each frame in the GIF in seconds.

    Returns:
    - None
    """
    # Prepare delta_heat_min and delta_heat_max if not provided
    if delta_heat_min is None or delta_heat_max is None:
        # Collect all positive incremental heat values across all time steps
        positive_delta_heats = np.concatenate([delta_heat[delta_heat > 0] for delta_heat in delta_heats])

        # Check if there are any positive incremental heat values
        if positive_delta_heats.size == 0:
            raise ValueError("All incremental heat values are zero or negative. Cannot normalize.")

        # Compute global min and max incremental heat values for normalization
        delta_heat_min = np.min(positive_delta_heats)
        delta_heat_max = np.max(positive_delta_heats)

    # Prepare for saving frames
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualization parameters
    cmap = plt.get_cmap(cmap)
    x_margin = 0.05 * (filtered_x.max() - filtered_x.min())
    y_margin = 0.05 * (filtered_y.max() - filtered_y.min())
    x_lim = (filtered_x.min() - x_margin, filtered_x.max() + x_margin)
    y_lim = (filtered_y.min() - y_margin, filtered_y.max() + y_margin)

    # Segments for edge plotting (static)
    if len(mapped_ind_i) != len(mapped_ind_j):
        raise ValueError("mapped_ind_i and mapped_ind_j must have the same length.")

    segments = np.array([
        [
            [filtered_x[mapped_ind_i[j]], filtered_y[mapped_ind_i[j]]],
            [filtered_x[mapped_ind_j[j]], filtered_y[mapped_ind_j[j]]]
        ]
        for j in range(len(mapped_ind_i))
    ])

    # Loop over time steps to create frames
    filenames = []
    for idx, (t, delta_heat) in enumerate(zip(time_steps, delta_heats)):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # Add edges as a LineCollection
        if len(segments) > 0:
            edge_collection = LineCollection(
                segments, colors='gray', linewidths=0.2, alpha=0.5
            )
            ax.add_collection(edge_collection)

        # Mask zero or negative incremental heat values
        delta_heat_masked = ma.masked_less_equal(delta_heat, 0)

        # Plot nodes with normalized color scale
        sc = ax.scatter(
            filtered_x, filtered_y, c=delta_heat_masked, cmap=cmap, s=1.8, edgecolor='none',
            norm=LogNorm(vmin=delta_heat_min, vmax=delta_heat_max)
        )

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_title(f'Incremental Heat at Time $t = {t:.2f}$', fontsize=10)
        ax.axis('equal')
        ax.axis('off')  # Turn off axis lines and labels for a cleaner view

        # Add a colorbar with smaller font sizes
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("New Heat Intensity", fontsize=8)  # Smaller colorbar label font size
        cbar.ax.tick_params(labelsize=6)  # Smaller colorbar tick font size

        # Save the frame
        filename = os.path.join(output_dir, f'frame_{idx:03d}.png')
        plt.savefig(filename, bbox_inches='tight')
        filenames.append(filename)

        plt.close(fig)  # Close the figure to free memory

    # Create a GIF from the saved frames
    with imageio.get_writer(gif_name, mode='I', duration=frame_duration, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Optionally, clean up the frames
    for filename in filenames:
        os.remove(filename)

    print(f"GIF saved as '{gif_name}'")

def plot_three_time_steps(time_steps, heat_distributions, target_indices, x, y, mapped_ind_i, mapped_ind_j, cmap='coolwarm', margin_factor=0.3):
    """
    Plots heat distributions at three target time steps in a horizontal multiplot, including edges
    within a square around maximal heat values on the first subplot. The color bar spans the entire plot
    placed underneath.

    Parameters:
    - time_steps (array-like): Array of time values for each heat distribution.
    - heat_distributions (list of arrays): List where each element is an array of heat values for a time step.
    - target_indices (list of int): List of three indices in heat_distributions to plot.
    - x, y (array-like): Coordinates of nodes in the network.
    - mapped_ind_i, mapped_ind_j (array-like): Indices of nodes defining edges.
    - cmap (str): Colormap for plotting heat distributions.
    - margin_factor (float): Factor for margin around the bounding square of maximal heat values.
    """
    if len(target_indices) != 3:
        raise ValueError("Please provide exactly three indices for target time steps.")
    
    # Determine global min and max for consistent color mapping across plots
    all_positive_heat_values = np.concatenate([
        heat_distributions[idx][heat_distributions[idx] > 0] for idx in target_indices
    ])
    vmin = all_positive_heat_values.min()
    vmax = all_positive_heat_values.max()
    
    # Step 1: Find bounds around maximal heat values for the first time step
    heat_first_step = heat_distributions[target_indices[0]]
    max_heat_idx = np.argmax(heat_first_step)
    x_max, y_max = x[max_heat_idx], y[max_heat_idx]
    
    # Define a square region around the maximal heat point
    x_margin = margin_factor * (x.max() - x.min()-12)
    y_margin = margin_factor * (y.max() - y.min()+10)
    x_bounds = (x_max - x_margin, x_max + x_margin)
    y_bounds = (y_max - y_margin, y_max + y_margin)
    
    # Step 2: Filter edges that are within this bounding box
    filtered_segments = [
        [
            [x[mapped_ind_i[j]], y[mapped_ind_i[j]]],
            [x[mapped_ind_j[j]], y[mapped_ind_j[j]]]
        ]
        for j in range(len(mapped_ind_i))
        if (x_bounds[0] <= x[mapped_ind_i[j]] <= x_bounds[1] and y_bounds[0] <= y[mapped_ind_i[j]] <= y_bounds[1]) or
           (x_bounds[0] <= x[mapped_ind_j[j]] <= x_bounds[1] and y_bounds[0] <= y[mapped_ind_j[j]] <= y_bounds[1])
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=500,gridspec_kw={'wspace': 0.05})
    
    for idx_ax, (ax, idx) in enumerate(zip(axes, target_indices)):
        heat = heat_distributions[idx]
        time = time_steps[idx]
        
        # Mask zero or negative values for consistent visualization
        heat_masked = ma.masked_less_equal(heat, 0)
        
        # Plot edges only on the first subplot (leftmost one) within the bounding box
        if idx_ax == 0 and len(filtered_segments) > 0:
            edge_collection = LineCollection(
                filtered_segments, colors='gray', linewidths=0.4, alpha=0.5
            )
            ax.add_collection(edge_collection)
            
        
        sc = ax.scatter(x, y, c=heat_masked, cmap=cmap, s=10, edgecolor='none',
                        norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f'Heat Distribution at $t = {time:.2f}$')
        ax.axis('equal')
        ax.axis('off')
    
    # Add a colorbar underneath the entire plot
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.15)
    cbar.set_label("Heat Intensity", fontsize=12)
    plt.show()