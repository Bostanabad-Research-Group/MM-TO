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

def eval_model_adjoint(model_list):
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = model_list[0].MP['P'][0]  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    E_tensor = model_list[0].MP['E'][0] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    wt = model_list[0].MP['wt'][0] # e.g., [1.0, 1.0, 1.0, 1.0] a tensor
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['pf']
    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    f_magnitude = model_list[0].f_magnitude
    f_index = model_list[0].f_index
    conn = model_list[0].conn
    B = model_list[0].B
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]
    
    for model in model_list:
        model.eval()

    m_phase = model_list[4].mean_module_NN_All(elem_x)

    if model_list[0].MP['design_flag']:
        g_phases = [model_list[i].covar_module(model_list[i].train_inputs[0], elem_x).evaluate() for i in range(4,4 + num_phase)]
    
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            for model in model_list[0:4]:
                model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()
            if model.MP['design_flag']:
                for model in model_list[4:4+num_phase]:
                    model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()

    
    if model_list[0].MP['design_flag']:
        K_inv_offsets = []
        for i in range(4, 4 + num_phase):  # phases corresponding to model_list[4] and model_list[5]
            phase_idx = i - 4  # 0 for model_list[4], 1 for model_list[5]
            model = model_list[i]
            residual = model.train_targets.unsqueeze(-1) - model_list[4].mean_module_NN_All(model.train_inputs[0])[:, phase_idx].unsqueeze(-1)
            K_inv_offsets.append(model.chol_decomp._cholesky_solve(residual))


    if model_list[0].MP['design_flag']:
        phases = [m_phase[:, i].unsqueeze(-1) + g_phases[i].t() @ K_inv_offsets[i] for i in range(len(K_inv_offsets))]
        phase_weights = torch.cat(phases, dim=1)
    else:
        phase_weights = m_phase
    
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    
    E = (E_tensor * phase_weights ** p).sum(dim = 1) # shape: [n_elem]

    elem_x = elem_x.detach().numpy()
    rho = phase_density.unsqueeze(-1).detach().numpy()
    Data = {'elem_x':elem_x,'rho':rho,'collocation_x':collocation_x,'conn':conn}
    return Data

def eval_model_adjoint_batch(model_list, batch_size=20000):
    """
    Evaluate phase densities (and E tensor) for large elem_x arrays in batches.
    """
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D'][0]          # e.g., [0, 0.4, 0.6, 1.0]
    P = model_list[0].MP['P'][0]          # e.g., [0, 1.6, 1.2, 1.0]
    E_tensor = model_list[0].MP['E'][0]   # e.g., [1e-9, 0.5, 0.7, 1.0]
    wt = model_list[0].MP['wt'][0]
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['pf']

    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    f_magnitude = model_list[0].f_magnitude
    f_index = model_list[0].f_index
    conn = model_list[0].conn
    B = model_list[0].B
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]

    # Ensure models are in eval mode
    for model in model_list:
        model.eval()

    # Precompute base mean for all phases
    # This will be done batch-wise later for memory safety
    design_flag = model_list[0].MP['design_flag']

    # Precompute cholesky decompositions if not yet available
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            # For displacement/stress models
            for model in model_list[0:4]:
                model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()
            # For GP phases
            if design_flag:
                for model in model_list[4:4+num_phase]:
                    model.chol_decomp = model.covar_module(model.train_inputs[0]).cholesky()

    # Compute inverse K residual offsets for GP phases
    if design_flag:
        K_inv_offsets = []
        for i in range(4, 4 + num_phase):
            phase_idx = i - 4
            model = model_list[i]
            residual = (
                model.train_targets.unsqueeze(-1)
                - model_list[4].mean_module_NN_All(model.train_inputs[0])[:, phase_idx].unsqueeze(-1)
            )
            K_inv_offsets.append(model.chol_decomp._cholesky_solve(residual))

    # --- Batched evaluation ---
    num_batches = int(np.ceil(N_elem_w / batch_size))
    phase_density_list = []
    E_list = []

    print(f"Evaluating {N_elem_w} elements in {num_batches} batches (batch size = {batch_size})")

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, N_elem_w)
        batch_x = elem_x[start:end, :]

        with torch.no_grad():
            # Mean prediction for all phases
            m_phase_batch = model_list[4].mean_module_NN_All(batch_x)

            # If design_flag enabled, apply GP correction per phase
            if design_flag:
                g_phases = [
                    model_list[i].covar_module(model_list[i].train_inputs[0], batch_x).evaluate()
                    for i in range(4, 4 + num_phase)
                ]
                phases = [
                    m_phase_batch[:, i].unsqueeze(-1) + g_phases[i].t() @ K_inv_offsets[i]
                    for i in range(len(K_inv_offsets))
                ]
                phase_weights = torch.cat(phases, dim=1)
            else:
                phase_weights = m_phase_batch

            # Compute density and E field for this batch
            phase_density_batch = (phase_weights * D).sum(dim=1)           # [batch_size]
            E_batch = (E_tensor * phase_weights ** p).sum(dim=1)           # [batch_size]

        phase_density_list.append(phase_density_batch)
        E_list.append(E_batch)

        print(f"  → Processed batch {b+1}/{num_batches}  ({end}/{N_elem_w} elements)")

    # Concatenate all batches
    phase_density = torch.cat(phase_density_list, dim=0)
    E_all = torch.cat(E_list, dim=0)

    # Prepare final outputs
    elem_x_np = elem_x.detach().cpu().numpy()
    rho_np = phase_density.unsqueeze(-1).detach().cpu().numpy()

    Data = {
        'elem_x': elem_x_np,
        'rho': rho_np,
        'E': E_all.detach().cpu().numpy(),
        'collocation_x': collocation_x,
        'conn': conn
    }

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

def extract_data_adjoint(save_folder, iter, mesh_key):
    # load the MP
    MP = torch.load(save_folder + 'MP.pt', map_location=torch.device('cpu'))
    base_folder = MP['base_folder']
    Example = MP['Example']
    N_worker = MP['N_worker']
    Diff_type = MP['Diff_type']
    num_phase = MP["num_phase"]

    # load the NN_config
    with open(save_folder + "NN_config_disp.json", "r") as file:
        NN_config_disp = json.load(file)
    with open(save_folder + "NN_config_rho.json", "r") as file:
        NN_config_rho = json.load(file)
    
    # load the training data:
    Training = torch.load(save_folder + "Training.pth",map_location=torch.device('cpu'))

    u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False)
    t1_X_train = Training['t1_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    t1_train = Training['t1_train'].type(tkwargs["dtype"]).requires_grad_(False)
    t2_X_train = Training['t2_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    t2_train = Training['t2_train'].type(tkwargs["dtype"]).requires_grad_(False)
    phase_train = Training['phase_train']
    
    model_u = LMGP(u_X_train, u_train, NN_config_disp,
                    name_output="u", MP=MP, num_output=2)
    model_v = LMGP(v_X_train, v_train, NN_config_disp,
                    name_output="v", MP=MP, num_output=2)
    model_t1 = LMGP(t1_X_train, t1_train, NN_config_disp,
                    name_output="t1", MP=MP, num_output=2)
    model_t2 = LMGP(t2_X_train, t2_train, NN_config_disp,
                    name_output="t2", MP=MP, num_output=2)
     # One GP model per phase
    model_phases = []
    for i in range(num_phase):
        phase_X, phase_y = phase_train[i]   # from your phase_train list
        model_phase_i = LMGP(phase_X, phase_y, NN_config_rho,
                                name_output=f"rho", MP=MP, num_output=num_phase)
        model_phases.append(model_phase_i)

    # Keep all in list for BC handling
    model_list = [model_u, model_v, model_t1, model_t2] + model_phases

    checkpoint = torch.load(save_folder + f'Trained_models_{iter}.pth',map_location=torch.device('cpu'))
    model_list[0].mean_module_NN_All.load_state_dict(checkpoint['model_u_state_dict'])
    model_list[2].mean_module_NN_All.load_state_dict(checkpoint['model_adj_state_dict'])
    model_list[4].mean_module_NN_All.load_state_dict(checkpoint['model_rho_state_dict'])
    
    # load mesh file:
    # file_loc = base_folder + 'Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    if mesh_key == 'Mesh_pred':
        file_loc = 'Data/' + f'{Example}_pred_{Diff_type}.pt'
    else:
        file_loc = 'Data/' + f'{Example}_GPU{1}_{Diff_type}.pt'
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
    model_list[0].collocation_x    = mesh_data['GPU0'][mesh_key]["X_node"]
    model_list[0].elem_x           = mesh_data['GPU0'][mesh_key]["X_elem"]
    model_list[0].elem_vol         = mesh_data['GPU0'][mesh_key]["elem_vol"]
    model_list[0].f_index          = mesh_data['GPU0'][mesh_key]["f_index"]
    model_list[0].f_magnitude      = mesh_data['GPU0'][mesh_key]["f_magnitude"]
    model_list[0].K_in_index       = mesh_data['GPU0'][mesh_key]["K_in_index"]
    model_list[0].K_in_magnitude   = mesh_data['GPU0'][mesh_key]["K_in_magnitude"]
    model_list[0].K_out_index      = mesh_data['GPU0'][mesh_key]["K_out_index"]
    model_list[0].K_out_magnitude  = mesh_data['GPU0'][mesh_key]["K_out_magnitude"]
    model_list[0].conn             = mesh_data['GPU0'][mesh_key]["conn"]
    model_list[0].B                = mesh_data['GPU0'][mesh_key]["B"]
    model_list[0].detJ             = mesh_data['GPU0'][mesh_key]["detJ"]

    # data = eval_model_adjoint(model_list)
    data = eval_model_adjoint_batch(model_list)
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

    pc = PatchCollection(
        patches,
        array=np.array(colors),
        cmap=cmap,
        edgecolor="none"
    )

    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title)

    cbar = plt.colorbar(pc, ax=ax, shrink=0.7)

    return pc, cbar

tkwargs = get_tkwargs()

save_folder = 'Results/EX2D2/multi_material/'
cmap_selected = cmap_phases
iter = [999,1999,2999,3999,4999,5999,6999,7999,8999]
iter = [9999]
# mesh_key = 'Mesh_51'
mesh_key = 'Mesh_pred'
run_num_list = [1,2,3,4,5,6,7,8,9,10]
for run_num in run_num_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')

    data_list = []
    for _,  i in enumerate(iter):
        # Extract data for each iteration
        data = extract_data_adjoint(run_folder, i, mesh_key)
        elem_x = data['elem_x']
        rho = data['rho']
        conn = data['conn']
        X_node = data['collocation_x']
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
        # plot density
        rho = np.clip(rho, 0.0, 1.0)
        fig_stress, axs_stress = plt.subplots(1, 1, figsize=(10, 6))
        pc, cbar = plot_field(axs_stress, conn, X_node, rho, r"$\rho$", cmap_selected)
        pc.set_rasterized(True)
        fig_stress.suptitle(f"Density Epoch{iter[i]}", fontsize=18)
        file_name = f"TO_evolution_density_{mesh_key}_Epoch{iter[i]}.pdf"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='pdf', dpi=600)

