import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from models.lmgp_updated import LMGP 
from utils.utils_general import get_tkwargs, projectDensity
import matplotlib.pyplot as plt
from gpytorch.settings import cholesky_jitter
from utils.utils_general import central_diff_2nd as ND2
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as mtri
from matplotlib.tri import Triangulation


colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_phase = ['#FFFFFF', '#33a645', '#3b4cc0', '#b40426'] 
Density = [0, 0.4, 0.6, 1.0]
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

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

def eval_model(model_list):
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = model_list[0].MP['P'][0]  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    kappa_tensor = model_list[0].MP['kappa'][0] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    wt = model_list[0].MP['wt'][0] # e.g., [1.0, 1.0, 1.0, 1.0] a tensor
    # nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    s = model_list[0].MP['s']
    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    # f_magnitude = model_list[0].f_magnitude
    # f_index = model_list[0].f_index
    conn = model_list[0].conn
    B = model_list[0].B
    detJ = model_list[0].detJ
    N = model_list[0].N
    N_elem_w = elem_x.shape[0]
    
    for model in model_list:
        model.eval()

    m_col = model_list[0].mean_module_NN_All(collocation_x)
    m_phase = model_list[1].mean_module_NN_All(elem_x)

    g_T = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()

    K_inv_offset_T = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))

    Node_T = (m_col[:,0].unsqueeze(-1) + g_T.t() @ K_inv_offset_T) #.squeeze(-1) # shape: (N_node_w, 2)
    # Node_disp = torch.cat([u, v], dim=1)  # shape: (N_node_w, 2)

    # Apply softmax across phase dimension (dim=1)
    phase_weights = m_phase
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]

    collocation_x = collocation_x.detach().numpy()
    Node_T = Node_T.detach().numpy()
    elem_x = elem_x.detach().numpy()
    rho = phase_density.unsqueeze(-1).detach().numpy()
    Data = {'collocation_x':collocation_x,'T':Node_T,'elem_x':elem_x,'rho':rho,'conn':conn}
    return Data

def eval_model_rho(model_list, batch_size=2000):
    """
    Evaluate phase densities for large elem_x in batches, with progress output.
    """
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D'][0]            # tensor, e.g. [0, 0.4, 0.6, 1.0]
    P = model_list[0].MP['P'][0]            # tensor, e.g. [0, 1.6, 1.2, 1.0]
    kappa_tensor = model_list[0].MP['kappa'][0]
    wt = model_list[0].MP['wt'][0]
    p = model_list[0].MP['p']
    s = model_list[0].MP['s']

    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    conn = model_list[0].conn
    B = model_list[0].B
    detJ = model_list[0].detJ
    N = model_list[0].N
    N_elem_w = elem_x.shape[0]

    # Ensure models in eval mode
    for model in model_list:
        model.eval()

    # Precompute covariance decomposition if not available
    if model_list[0].chol_decomp is None:
        from gpytorch.settings import cholesky_jitter
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(
                model_list[0].train_inputs[0]
            ).cholesky()

    # --- Batched evaluation ---
    num_batches = int(np.ceil(N_elem_w / batch_size))
    phase_density_list = []

    print(f"Evaluating {N_elem_w} elements in {num_batches} batches (batch size = {batch_size})")

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, N_elem_w)
        batch_x = elem_x[start:end, :]

        # Evaluate model prediction for this batch
        with torch.no_grad():
            m_phase_batch = model_list[1].mean_module_NN_All(batch_x)

        # Compute density for this batch
        phase_density_batch = (m_phase_batch * D).sum(dim=1)  # [batch_size]

        phase_density_list.append(phase_density_batch)

        # Print progress
        print(f"  → Processed batch {b+1}/{num_batches}  ({end}/{N_elem_w} elements)")

    # Concatenate all batches
    phase_density = torch.cat(phase_density_list, dim=0)

    # Prepare final outputs
    elem_x_np = elem_x.detach().cpu().numpy()
    rho_np = phase_density.unsqueeze(-1).detach().cpu().numpy()

    Data = {'elem_x': elem_x_np, 'rho': rho_np}
    print("✅ Evaluation completed successfully.")
    return Data

def moving_average(arr, window_size):
    """
    Apply a simple moving average filter to smooth the input array.
    
    Parameters:
    - arr: numpy array, the input data to be smoothed.
    - window_size: int, the size of the moving window for averaging.
    
    Returns:
    - smoothed: numpy array, the smoothed data.
    """
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def extract_data(save_folder, iter, mesh_key):
    # load the MP
    MP = torch.load(save_folder + 'MP.pt', map_location=torch.device('cpu'))
    base_folder = MP['base_folder']
    Example = MP['Example']
    N_worker = MP['N_worker']
    Diff_type = MP['Diff_type']

    # load the NN_config
    with open(save_folder + "NN_config_disp.json", "r") as file:
        NN_config_disp = json.load(file)
    with open(save_folder + "NN_config_rho.json", "r") as file:
        NN_config_rho = json.load(file)
    
    # load the training data:
    Training = torch.load(save_folder + "Training.pth",map_location=torch.device('cpu'))

    T_X_train = Training['T_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    T_train = Training['T_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False)
    
    model_u = LMGP(T_X_train, T_train, NN_config_disp,
                    name_output="u", MP=MP, num_output=1)
    model_phase = LMGP(rho_X_train, rho_train, NN_config_rho,
                        name_output="rho", MP=MP, num_output=MP['num_phase'])


    model_list = [model_u, model_phase]
    checkpoint = torch.load(save_folder + f'Trained_models_{iter}.pth',map_location=torch.device('cpu'))
    model_list[0].mean_module_NN_All.load_state_dict(checkpoint['model_u_state_dict'])
    model_list[1].mean_module_NN_All.load_state_dict(checkpoint['model_rho_state_dict'])
    
    # load mesh file:
    # file_loc = base_folder + 'Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    file_loc = 'Data/' + f'{Example}_GPU{1}_{Diff_type}.pt'
    if mesh_key == 'Mesh_pred':
        file_loc = 'Data/' + f'{Example}_pred_CPS4_reduced.pt'
    mesh_data = torch.load(file_loc)

    # mesh_key = 'Mesh_50'
    # X_elem0 = mesh_data['GPU0'][mesh_key]['X_elem']
    # X_elem1 = mesh_data['GPU1'][mesh_key]['X_elem']
    # X_node0 = mesh_data['GPU0'][mesh_key]['X_node']
    # X_node1 = mesh_data['GPU1'][mesh_key]['X_node']

    # # --- Scatter plot ---
    # plt.figure(figsize=(12, 12))
    # plt.scatter(X_elem0[:, 0], X_elem0[:, 1], 
    #             s=1, c='red', label='GPU0:elem')
    # plt.scatter(X_node0[:, 0], X_node0[:, 1], 
    #             s=4, c='orange', label='GPU0:node')
    # plt.scatter(X_elem1[:, 0], X_elem1[:, 1], 
    #             s=1, c='blue', label='GPU1:elem')
    # plt.scatter(X_node1[:, 0], X_node1[:, 1], 
    #             s=1, c='cyan', label='GPU1:node')

    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.axis("equal")
    # plt.title("Element partition: GPU0 (red) vs GPU1 (blue)")
    # plt.show()
    
    # Save all data in a dictionary
    # assign mesh data to model_u
    GPU_key = 'GPU0'
    model_list[0].collocation_x = mesh_data[GPU_key][mesh_key]["X_node"]
    model_list[0].elem_x        = mesh_data[GPU_key][mesh_key]["X_elem"]
    model_list[0].elem_vol        = mesh_data[GPU_key][mesh_key]["elem_vol"]
    model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
    model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
    model_list[0].N             = mesh_data[GPU_key][mesh_key]["N"]
    model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]

    data = eval_model_rho(model_list)
    return data

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

def plot_field(ax, conn, X_node, field, title, cmap):
    patches, colors = [], []
    for e, nodes in enumerate(conn):
        poly_coords = X_node[nodes]
        patches.append(Polygon(poly_coords, closed=True))
        colors.append(field[e].item() if field.ndim == 1 else field[e].mean().item())

    pc = PatchCollection(patches, array=np.array(colors),
                         cmap=cmap, edgecolor="none")
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(pc, ax=ax, shrink=0.7)

tkwargs = get_tkwargs()

save_folder = 'Results/EX2D1/multi_material/'
cmap_selected = cmap_phases

iter = [9999]
# mesh_key = 'Mesh_pred'
mesh_key = 'Mesh_01'
# run_num_list = [1,2,3]
run_num_list = [1,2,3,4,5,6,7,8,9,10]
for run_num in run_num_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')

    data_list = []
    for _,  i in enumerate(iter):
        # Extract data for each iteration
        data = extract_data(run_folder, i, mesh_key)
        elem_x = data['elem_x']
        rho = data['rho']
        data_list.append({
            'elem_x': elem_x,
            'rho':rho,
        })

    blue_dark = hex_to_rgba(cmap_curves['blue_dark'])
    blue_light = hex_to_rgba(cmap_curves['blue_light'])
    red_dark = hex_to_rgba(cmap_curves['red_dark'])
    red_light = hex_to_rgba(cmap_curves['red_light'])
    green_dark = hex_to_rgba(cmap_curves['green_dark'])
    green_light = hex_to_rgba(cmap_curves['green_light'])
    cyan_dark = hex_to_rgba(cmap_curves['cyan_dark'])
    cyan_light = hex_to_rgba(cmap_curves['cyan_light'])
    magenta_dark = hex_to_rgba(cmap_curves['magenta_dark'])
    magenta_light = hex_to_rgba(cmap_curves['magenta_light'])
    yellow_dark = hex_to_rgba(cmap_curves['yellow_dark'])
    yellow_light = hex_to_rgba(cmap_curves['yellow_light'])

    # plot parameters
    font = "Times New Roman"
    fontsize_title = 20
    fontsize_anno = 20
    fontsize_label = 18
    fontsize_ticks = 18
    fontsize_legend = 18
    anno_loc = [-0.15,0.99]

    box_linewidth = 1.5
    legend_linewidth = 1
    colorbar_shrink = 0.6
    curve_linewidth = 2

    # Create the plot with size (8, 6)
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': font,
        'axes.labelsize': fontsize_label,
        'axes.titlesize': fontsize_title,
        'xtick.labelsize': fontsize_ticks,
        'ytick.labelsize': fontsize_ticks,
        'legend.fontsize': fontsize_legend, 
    })
    plt.rcParams["mathtext.fontset"] = "stix"

    # --- Generate plots ---
    for i, data in enumerate(data_list):
        elem_x = data['elem_x']          # shape: [n_elem, 2]
        rho = data['rho'].flatten()      # shape: [n_elem]

        fig, ax = plt.subplots(figsize=(10, 6))
        num_levels = 500

        # --- Contour plot using triangulation (no xi, yi needed) ---
        tri = Triangulation(elem_x[:, 0], elem_x[:, 1])
        contour = ax.tricontourf(
            tri, rho, levels=num_levels,
            cmap=cmap_selected, vmin=0, vmax=1
        )

        # --- Axis formatting ---
        ax.set_aspect('equal')
        ax.set_rasterized(True)
        ax.set_title(f"Epoch: {iter[i]}", fontsize=fontsize_title, pad=10)
        ax.axis('off')

        # --- Colorbar (auto placement, no overlap) ---
        cbar = fig.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_ticks(np.arange(0.0, 1.0001, 0.2))
        cbar.ax.tick_params(labelsize=fontsize_ticks)
        cbar.outline.set_linewidth(1.0)

        # --- Layout and save ---
        plt.tight_layout(pad=0.5)
        file_name = f"TO_evolution_density_{mesh_key}_Epoch{iter[i]}.pdf"
        file_path = os.path.join(run_folder, file_name)
        plt.savefig(file_path, format='pdf', dpi=600, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Saved: {file_name}")

        print(f"✅ Saved density field for Epoch {iter[i]} → Run_{run_num}")



