import os
import scipy.io
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb

from matplotlib.cm import ScalarMappable
import pyvista as pv

colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_phase = ['#FFFFFF', '#33a645', '#3b4cc0', '#b40426'] 
Density = [0, 0.4, 0.6, 1.0]
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

# --------- Function to save colormap ----------
def save_colormap_to_xml(cmap, filename, n_samples=256):
    x = np.linspace(0, 1, n_samples)
    rgb_array = cmap(x)

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<ColorMaps>\n')
        f.write(f'  <ColorMap name="{os.path.splitext(os.path.basename(filename))[0]}" space="RGB">\n')
        for xi, rgba in zip(x, rgb_array):
            r, g, b = rgba[:3]
            f.write(f'    <Point x="{xi:.6f}" r="{r:.6f}" g="{g:.6f}" b="{b:.6f}" o="1.0"/>\n')
        f.write('  </ColorMap>\n')
        f.write('</ColorMaps>\n')
    print(f"[✔] Saved: {filename} (import into ParaView)")

# --------- Save both colormaps ----------
save_colormap_to_xml(cmap_phases, "cmap_phases.xml")
save_colormap_to_xml(cmap_density, "cmap_density.xml")

# Sample N uniformly spaced RGB values
n_samples = 256
x = np.linspace(0, 1, n_samples)
rgb_array = cmap_phases(x)

# Write to ParaView XML colormap file
with open("cmap_phases.xml", "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<ColorMaps>\n')
    f.write('  <ColorMap name="cmap_phases" space="RGB">\n')
    for xi, rgba in zip(x, rgb_array):
        r, g, b = rgba[:3]
        f.write(f'    <Point x="{xi:.6f}" r="{r:.6f}" g="{g:.6f}" b="{b:.6f}" o="1.0"/>\n')
    f.write('  </ColorMap>\n')
    f.write('</ColorMaps>\n')

print("[✔] Saved: cmap_phases.xml (import into ParaView)")

cmap_phases_disc = ListedColormap(colors_phase, name='cmap_phases_disc')
with open("cmap_phases_disc.xml", "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<ColorMaps>\n')
    f.write('  <ColorMap name="cmap_phases_disc" space="RGB" indexedLookup="true">\n')
    
    for i, hex_color in enumerate(colors_phase):
        r, g, b = to_rgb(hex_color)
        f.write(f'    <Point x="{i}" r="{r:.6f}" g="{g:.6f}" b="{b:.6f}" o="1.0"/>\n')

    f.write('  </ColorMap>\n')
    f.write('</ColorMaps>\n')

colors_contours = ["#3b4fbf", "#79a2f2", "#c2d7f2", "#f2c9bb", "#f2856d", "#bf0637"]
cmap_contours = LinearSegmentedColormap.from_list("custom_cmap", colors_contours, N=128)

cmap_curves = {'blue_dark': "#3b4cc0",'blue_light': "#8db0fe",
                 'red_dark':"#b40426",'red_light':"#f4987a",
                 'green_dark': "#33a645",'green_light': "#afd991",
                 'gray_dark': "#555555",'gray_light': "#bababa",
                 'purple_dark': "#6659a7",'purple_light': "#c2bdde",
                 'orange_dark': "#ef7e21",'orange_light': "#f6c092",
                 'yellow_dark': "#f6d31b",'yellow_light': "#fae98f",
                 'cyan_dark': "#7bddf3",'cyan_light': "#bbedf2",
                 'magenta_dark': "#fe3da7",'magenta_light': "#fe9fd3",
                 }

import numpy as np
import pyvista as pv

def export_density_to_vti(X_col_elem, rho, nelx, nely, nelz, spacing, filename='density.vti', preview=True):
    import numpy as np
    import pyvista as pv

    dx, dy, dz = spacing
    dims = (nelx + 1, nely + 1, nelz + 1)

    # Reshape rho directly
    rho_3d = rho.reshape((nelx, nely, nelz))

    # Compute origin
    origin = (
        X_col_elem[:, 0].min() - dx / 2,
        X_col_elem[:, 1].min() - dy / 2,
        X_col_elem[:, 2].min() - dz / 2
    )

    # Build the grid
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.spacing = (dx, dy, dz)
    grid.origin = origin
    grid.cell_data['density'] = rho_3d.flatten(order = 'F')
    
    print("Density range in grid:", grid.get_data_range())  # Should be ~[0, 1]

    # Optional visualization before saving
    # if preview:
    #     print("Previewing density volume...")
    #     grid.plot(
    #         volume=True,
    #         cmap=cmap_phases,
    #         opacity='sigmoid',
    #         scalar_bar_args={"title": "Density"},
    #         show_bounds=True
    #     )

    # Save .vti
    grid.save(filename)
    print(f"[✔] Saved: {filename}")

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert hex color to RGBA format and adjust alpha (transparency).
    Default alpha is 1.0 (fully opaque).
    
    Parameters:
    - hex_color: str, color in hex format (e.g., '#3b4cc0')
    - alpha: float, transparency level (0.0 to 1.0)
    
    Returns:
    - rgba: tuple, (r, g, b, a)
    """
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def scatter_plot_solid_voxels1(X_col_elem, rho, cmap_phases, title='', ax=None):
    """
    Scatter plot of voxels with rho > 0.9.
    Adds alpha (transparency) based on rho.
    All axes, ticks, labels, gridlines, and panes are hidden.
    """
    # Mask solid voxels
    mask = rho > 0.0
    X_solid = X_col_elem[mask]
    rho_solid = rho[mask]

    if len(X_solid) == 0:
        print("No voxels with rho > 0.9 found.")
        return None

    # Custom colormap
    norm = Normalize(vmin=0.0, vmax=1.0)

    # RGBA with nonlinear alpha
    rgba = cmap_phases(norm(rho_solid))
    # method 1
    rgba[:, -1] = rho_solid ** 2  # alpha: rho=0.9 → ~0.81, rho=1 → 1.0
    
    # # method 2
    # rho_min = 0.2
    # alpha = np.clip((rho_solid - rho_min) / (1.0 - rho_min), 0.0, 1.0)
    # rgba[:, -1] = alpha**2.5
    
    # Create plot
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_solid[:, 0], X_solid[:, 1], X_solid[:, 2],
               c=rgba, s=20, edgecolors='none', linewidth=0)

    # Equal aspect ratio
    ax.set_box_aspect([
        np.ptp(X_col_elem[:, 0]),
        np.ptp(X_col_elem[:, 1]),
        np.ptp(X_col_elem[:, 2])
    ])

    # 🔻 Remove visual clutter
    ax.set_axis_off()  # removes labels, ticks, and panes
    ax.grid(False)

    return ax

def scatter_plot_solid_voxels2(X_col_elem, rho, cmap_phases, rho_min, title='', ax=None):
    """
    Scatter plot of voxels with rho > 0.9.
    Adds alpha (transparency) based on rho.
    All axes, ticks, labels, gridlines, and panes are hidden.
    """
    # Mask solid voxels
    mask = rho > 0.0
    X_solid = X_col_elem[mask]
    rho_solid = rho[mask]

    if len(X_solid) == 0:
        print("No voxels with rho > 0.9 found.")
        return None

    # Custom colormap
    norm = Normalize(vmin=0.0, vmax=1.0)

    # RGBA with nonlinear alpha
    rgba = cmap_phases(norm(rho_solid))
    alpha = np.clip((rho_solid - rho_min) / (1.0 - rho_min), 0.0, 1.0)
    rgba[:, -1] = alpha**2.5
    
    # Create plot
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_solid[:, 0], X_solid[:, 1], X_solid[:, 2],
               c=rgba, s=20, edgecolors='none', linewidth=0)

    # Equal aspect ratio
    ax.set_box_aspect([
        np.ptp(X_col_elem[:, 0]),
        np.ptp(X_col_elem[:, 1]),
        np.ptp(X_col_elem[:, 2])
    ])

    # 🔻 Remove visual clutter
    ax.set_axis_off()  # removes labels, ticks, and panes
    ax.grid(False)

    return ax

def rotation_matrix_x(theta):
    """
    Rotation matrix around the X-axis by angle theta (in radians).
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

# Set global font and LaTeX settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'axes.labelsize': 20,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 36,
    'text.usetex': False,
})
linewidth = 2
anno_fontsize = 20

# save_folder = 'Results/EX3D3/mass_cost_long/'
# # iters = [999,1999,2999,3999,4999,9999]
# iters = list(range(24, 10000, 50))
# iters.append(9999)

# # define rho_min
# n_total = len(iters)          # 201
# n_ramp = 150
# rho_min_max = 0.15

# rho_min = np.zeros(n_total)
# rho_min[:n_ramp] = np.linspace(0.0, rho_min_max, n_ramp)
# rho_min[n_ramp:] = rho_min_max

# last_finished_iter = 6924
# iters = [i for i in iters if i > last_finished_iter]

# run_num_list = [4]
# theta_deg = 90
# symmetric = {'x':True,'y':False, 'z': False}
# for run_num in run_num_list:
#     run_folder = os.path.join(save_folder, f'Run_{run_num}/')
#     # load the MP
#     MP = torch.load(run_folder + 'MP.pt', map_location=torch.device('cpu'))
#     domain = MP['domain']
#     xmin, xmax,ymin,ymax,zmin,zmax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1], domain['z'][0], domain['z'][1],
#     if symmetric['x']:
#         xmax = 2*xmax
#     if symmetric['y']:
#         ymax = 2*ymax
#     if symmetric['z']:
#         zmax = 2*zmax

#     count = 0
#     for i in iters:
#         # filename = f'rho_{i}_update.mat'  # Correct string formatting
#         filename = f'rho_{i}.mat'  # Correct string formatting
#         filepath = os.path.join(run_folder, filename)

#         # Load the .mat file
#         mat_data = scipy.io.loadmat(filepath)
        
#         # Assuming the variable name inside is 'rho'
#         rho = mat_data['rho'] # Adjust this to match the actual variable name in the .mat file
#         rho = rho.transpose(1,0,2)
#         rho = np.clip(rho, 0.0, 1.0)
#         print(rho.shape)

#         # # plot the distribution of rho
#         # rho_flat = rho.flatten()  # or rho.ravel()
#         # plt.figure(figsize=(6, 4))
#         # plt.hist(rho_flat, bins=400, color='steelblue', edgecolor='black', density=True)
#         # plt.xlabel('Density')
#         # plt.ylabel('Probability Density')
#         # plt.grid(True, linestyle='--', alpha=0.5)
#         # plt.tight_layout()
#         # # plt.show()
#         # # Save the plot
#         # file_name = f"distribution_rho_{i}.jpeg"
#         # file_path = f"{run_folder}/{file_name}"
#         # plt.savefig(file_path, format='jpeg', dpi=600)

#         # create X_col_elem
#         Nelx, Nely, Nelz = rho.shape
#         xi = np.linspace(xmin, xmax, num=Nelx+1)
#         yi = np.linspace(ymin, ymax, num=Nely+1)
#         zi = np.linspace(zmin, zmax, num=Nelz+1)
#         dx = xi[1] - xi[0]
#         dy = yi[1] - yi[0]
#         dz = zi[1] - zi[0]
#         xi, yi, zi = np.meshgrid(xi, yi, zi)
#         xi = xi.transpose(1,0,2)
#         yi = yi.transpose(1,0,2)
#         zi = zi.transpose(1,0,2)
#         edge_length = [dx,dy,dz]
#         xi_elem = xi + dx/2.0
#         xi_elem = xi_elem[0:-1,0:-1,0:-1]
#         yi_elem = yi + dy/2.0
#         yi_elem = yi_elem[0:-1,0:-1,0:-1]
#         zi_elem = zi + dz/2.0
#         zi_elem = zi_elem[0:-1,0:-1,0:-1]
#         X_col_elem = np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T
#         # print(xi_elem.shape)
#         # print(X_col_elem.shape)

#         # visualie the density in 3D:
#         rho_flat = rho.flatten()

#         # Apply rotation: X_col_elem is (N, 3); Q is (3, 3)
#         theta_rad = np.radians(theta_deg)
#         Q = rotation_matrix_x(theta_rad)
#         X_col_elem = X_col_elem @ Q.T  # transpose Q to match dimensions

#         export_density_to_vti(X_col_elem, rho_flat, Nelx, Nely, Nelz,spacing=(dx, dy, dz), filename= run_folder + f'density_{i}.vti')

#         # Plot
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         scatter_plot_solid_voxels2(X_col_elem, rho_flat, cmap_phases, rho_min[count], title='Solid Structure', ax=ax)
#         count = count + 1
#         norm = Normalize(vmin=0.0, vmax=1.0)
#         sm = ScalarMappable(norm=norm, cmap=cmap_phases)
#         sm.set_array([])  # Dummy array required
#         cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
#         cbar.set_label(' ', rotation=270, labelpad=30,fontsize = 20)

#         ax.view_init(elev=20, azim=-50)
#         plt.tight_layout()
        
#         # Save the plot
#         file_name = f"rho_{i}.jpeg"
#         file_path = f"{run_folder}/{file_name}"
#         plt.savefig(file_path, format='jpeg', dpi=600)
#         # plt.show()


save_folder = 'Results/EX3D1/single_material/'
iters = [999,1999,2999,3999,4999,9999]
iters = [9999]
theta_deg = 0
run_num_list = [1,2,3,4,5,6,7,8,9,10]
run_num_list = [1]

cmap_selected = cmap_density # cmap_phases cmap_density

for run_num in run_num_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')
    # load the MP
    MP = torch.load(run_folder + 'MP.pt', map_location=torch.device('cpu'))
    domain = MP['domain']
    xmin, xmax,ymin,ymax,zmin,zmax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1], domain['z'][0], domain['z'][1],

    count = 0
    for i in iters:
        # filename = f'rho_{i}_update.mat'  # Correct string formatting
        filename = f'rho_{i}.mat'  # Correct string formatting
        filepath = os.path.join(run_folder, filename)

        # Load the .mat file
        mat_data = scipy.io.loadmat(filepath)
        
        # Assuming the variable name inside is 'rho'
        rho = mat_data['rho'] # Adjust this to match the actual variable name in the .mat file
        rho = rho.transpose(1,0,2)
        rho = np.clip(rho, 0.0, 1.0)
        # rho = np.flip(rho, axis=1)
        print(rho.shape)

        # # plot the distribution of rho
        # rho_flat = rho.flatten()  # or rho.ravel()
        # plt.figure(figsize=(6, 4))
        # plt.hist(rho_flat, bins=400, color='steelblue', edgecolor='black', density=True)
        # plt.xlabel('Density')
        # plt.ylabel('Probability Density')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # # plt.show()
        # # Save the plot
        # file_name = f"distribution_rho_{i}.jpeg"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='jpeg', dpi=600)

        # create X_col_elem
        Nelx, Nely, Nelz = rho.shape
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        edge_length = [dx,dy,dz]
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        X_col_elem = np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T
        # print(xi_elem.shape)
        # print(X_col_elem.shape)

        # visualie the density in 3D:
        rho_flat = rho.flatten()

        # Apply rotation: X_col_elem is (N, 3); Q is (3, 3)
        theta_rad = np.radians(theta_deg)
        Q = rotation_matrix_x(theta_rad)
        X_col_elem = X_col_elem @ Q.T  # transpose Q to match dimensions

        export_density_to_vti(X_col_elem, rho_flat, Nelx, Nely, Nelz,spacing=(dx, dy, dz), filename= run_folder + f'density_{i}.vti')

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter_plot_solid_voxels1(X_col_elem, rho_flat, cmap_selected, title='Solid Structure', ax=ax)
        count = count + 1
        norm = Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(norm=norm, cmap = cmap_selected)
        sm.set_array([])  # Dummy array required
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label(' ', rotation=270, labelpad=30,fontsize = 20)
        cbar.ax.tick_params(labelsize=36)   # increase tick font size
        ax.view_init(elev=20, azim=-50)
        plt.tight_layout()
        
        # Save the plot
        file_name = f"rho_{i}.jpeg"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='jpeg', dpi=600)
        # plt.show()




save_folder = 'Results/EX3D1/multi_material/'
iters = [999,1999,2999,3999,4999,9999]
iters = [9999]
theta_deg = 0
run_num_list = [1,2,3,4,5,6,7,8,9,10]
# run_num_list = [1,2,3]

cmap_selected = cmap_phases # cmap_phases cmap_density

for run_num in run_num_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')
    # load the MP
    MP = torch.load(run_folder + 'MP.pt', map_location=torch.device('cpu'))
    domain = MP['domain']
    xmin, xmax,ymin,ymax,zmin,zmax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1], domain['z'][0], domain['z'][1],

    count = 0
    for i in iters:
        # filename = f'rho_{i}_update.mat'  # Correct string formatting
        filename = f'rho_{i}.mat'  # Correct string formatting
        filepath = os.path.join(run_folder, filename)

        # Load the .mat file
        mat_data = scipy.io.loadmat(filepath)
        
        # Assuming the variable name inside is 'rho'
        rho = mat_data['rho'] # Adjust this to match the actual variable name in the .mat file
        rho = rho.transpose(1,0,2)
        rho = np.clip(rho, 0.0, 1.0)
        # rho = np.flip(rho, axis=1)
        print(rho.shape)

        # # plot the distribution of rho
        # rho_flat = rho.flatten()  # or rho.ravel()
        # plt.figure(figsize=(6, 4))
        # plt.hist(rho_flat, bins=400, color='steelblue', edgecolor='black', density=True)
        # plt.xlabel('Density')
        # plt.ylabel('Probability Density')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # # plt.show()
        # # Save the plot
        # file_name = f"distribution_rho_{i}.jpeg"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='jpeg', dpi=600)

        # create X_col_elem
        Nelx, Nely, Nelz = rho.shape
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        edge_length = [dx,dy,dz]
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        X_col_elem = np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T
        # print(xi_elem.shape)
        # print(X_col_elem.shape)

        # visualie the density in 3D:
        rho_flat = rho.flatten()

        # Apply rotation: X_col_elem is (N, 3); Q is (3, 3)
        theta_rad = np.radians(theta_deg)
        Q = rotation_matrix_x(theta_rad)
        X_col_elem = X_col_elem @ Q.T  # transpose Q to match dimensions

        export_density_to_vti(X_col_elem, rho_flat, Nelx, Nely, Nelz,spacing=(dx, dy, dz), filename= run_folder + f'density_{i}.vti')

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter_plot_solid_voxels1(X_col_elem, rho_flat, cmap_selected, title='Solid Structure', ax=ax)
        count = count + 1
        norm = Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(norm=norm, cmap = cmap_selected)
        sm.set_array([])  # Dummy array required
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label(' ', rotation=270, labelpad=30,fontsize = 20)
        cbar.ax.tick_params(labelsize=36)   # increase tick font size

        ax.view_init(elev=20, azim=-50)
        plt.tight_layout()
        
        # Save the plot
        file_name = f"rho_{i}.jpeg"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='jpeg', dpi=600)
        # # plt.show()



