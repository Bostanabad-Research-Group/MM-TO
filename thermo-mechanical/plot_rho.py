import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from models.lmgp_updated3 import LMGP 
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

colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_phase = ['#FFFFFF', '#33a645', '#3b4cc0', '#b40426'] 
Density = [0, 0.4, 0.6, 1.0]
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

Density = np.array([0, 2.7, 7.8, 8.96])  # density g/cm^3: Ni, steel, Al, ti, Copper: 8.9, 7.8, 2.7, 8.96, 4.5
Density = Density/Density.max()
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases_AlCuFe = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

Density = np.array([0, 4.5, 7.8, 8.96])  # density g/cm^3: Ni, steel, Al, ti, Copper: 8.9, 7.8, 2.7, 8.96, 4.5
Density = Density/Density.max()
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases_TiCuFe = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)


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
    alpha_tensor = model_list[0].MP['alpha'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    kappa_tensor = model_list[0].MP['kappa'][0]
    deltaT = model_list[0].MP['deltaT']
    D = model_list[0].MP['D'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = model_list[0].MP['P'][0]  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    E_tensor = model_list[0].MP['E'][0] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    wt = model_list[0].MP['wt'][0] # e.g., [1.0, 1.0, 1.0, 1.0] a tensor
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    # f_magnitude = model_list[0].f_magnitude
    # f_index = model_list[0].f_index
    conn = model_list[0].conn
    B = model_list[0].B
    B_T = model_list[0].B_T
    N = model_list[0].N
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]
    T_inf = model_list[0].MP['T_inf']
    thickness = model_list[0].MP['domain']['thickness']
    
    for model in model_list:
        model.eval()

    m_col_disp = model_list[0].mean_module_NN_All(collocation_x)
    m_col_disp_adj = model_list[1].mean_module_NN_All(collocation_x)
    m_col_T = model_list[2].mean_module_NN_All(collocation_x)
    m_col_T_adj = model_list[3].mean_module_NN_All(collocation_x)
    m_elem_phase = model_list[4].mean_module_NN_All(elem_x)

    # --- Displacement GP ---
    g_u = model_list[0].independent_kernels[0](model_list[0].train_inputs_per_output[0], collocation_x).evaluate()
    g_v = model_list[0].independent_kernels[1](model_list[0].train_inputs_per_output[1], collocation_x).evaluate()

    # --- Adjoint GP ---
    g_vd1 = model_list[1].independent_kernels[0](model_list[1].train_inputs_per_output[0], collocation_x).evaluate()
    g_vd2 = model_list[1].independent_kernels[1](model_list[1].train_inputs_per_output[1], collocation_x).evaluate()

    # --- T GP ---
    g_t = model_list[2].independent_kernels[0](model_list[2].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # --- T adjoint GP ---
    g_vt = model_list[3].independent_kernels[0](model_list[3].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # --- Phase GPs (only if design_flag = True) ---
    if model_list[0].MP['design_flag']:
        g_phases = []
        for i in range(model_list[4].num_output):
            k_i = model_list[4].independent_kernels[i]
            X_i = model_list[4].train_inputs_per_output[i]
            g_phases.append(k_i(X_i, elem_x).evaluate().detach())
    
    # ====================================================
    # Compute Cholesky decompositions and offsets
    # ====================================================
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            # --- Displacement kernels ---
            model_list[0].chol_decomp = []
            for i in range(model_list[0].num_output):
                K_i = model_list[0].independent_kernels[i](model_list[0].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[0].chol_decomp.append(L_i)

            # --- Displacement adjoint kernels ---
            model_list[1].chol_decomp = []
            for i in range(model_list[1].num_output):
                K_i = model_list[1].independent_kernels[i](model_list[1].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[1].chol_decomp.append(L_i)
            
            # --- T kernels ---
            model_list[2].chol_decomp = []
            for i in range(model_list[2].num_output):
                K_i = model_list[2].independent_kernels[i](model_list[2].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[2].chol_decomp.append(L_i)
                
            # --- T kernels ---
            model_list[3].chol_decomp = []
            for i in range(model_list[3].num_output):
                K_i = model_list[3].independent_kernels[i](model_list[3].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[3].chol_decomp.append(L_i)

            # --- Phase kernels (only if needed) ---
            if model_list[0].MP['design_flag']:
                model_list[4].chol_decomp = []
                for i in range(model_list[4].num_output):
                    K_i = model_list[4].independent_kernels[i](model_list[4].train_inputs_per_output[i]).evaluate().detach()
                    L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                    model_list[4].chol_decomp.append(L_i)

    # ====================================================
    # Compute K⁻¹·(y − μ)
    # ====================================================

    # --- Displacement ---
    K_inv_offset_u = torch.cholesky_solve(
        model_list[0].train_target_per_output[0].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[0].chol_decomp[0]
    )

    K_inv_offset_v = torch.cholesky_solve(
        model_list[0].train_target_per_output[1].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
        model_list[0].chol_decomp[1]
    )

    # --- Displacement Adjoint ---
    K_inv_offset_vd1 = torch.cholesky_solve(
        model_list[1].train_target_per_output[0].unsqueeze(-1)
        - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[1].chol_decomp[0]
    )

    K_inv_offset_vd2 = torch.cholesky_solve(
        model_list[1].train_target_per_output[1].unsqueeze(-1)
        - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
        model_list[1].chol_decomp[1]
    )

    # --- Temperature ---
    K_inv_offset_T = torch.cholesky_solve(
        model_list[2].train_target_per_output[0].unsqueeze(-1)
        - model_list[2].mean_module_NN_All(model_list[2].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[2].chol_decomp[0]
    )
    # --- Temperature Adjoint---
    K_inv_offset_T_adj = torch.cholesky_solve(
        model_list[3].train_target_per_output[0].unsqueeze(-1)
        - model_list[3].mean_module_NN_All(model_list[3].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[3].chol_decomp[0]
    )

    # --- Phase (if design_flag = True) ---
    if model_list[0].MP['design_flag']:
        K_inv_offsets_phase = []
        for i in range(model_list[4].num_output):
            y_i = model_list[4].train_target_per_output[i]
            X_i = model_list[4].train_inputs_per_output[i]
            mu_i = model_list[4].mean_module_NN_All(X_i)[:, i].unsqueeze(-1)
            offset_i = y_i.unsqueeze(-1) - mu_i
            K_inv_offsets_phase.append(
                torch.cholesky_solve(offset_i, model_list[4].chol_decomp[i])
            )

    u = (m_col_disp[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u)#.squeeze(-1)
    v = (m_col_disp[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v)#.squeeze(-1)
    vd1 = (m_col_disp_adj[:,0].unsqueeze(-1) + g_vd1.t() @ K_inv_offset_vd1)#.squeeze(-1)
    vd2 = (m_col_disp_adj[:,1].unsqueeze(-1) + g_vd2.t() @ K_inv_offset_vd2)#.squeeze(-1)
    Node_T = (m_col_T[:,0].unsqueeze(-1) + g_t.t() @ K_inv_offset_T)#.squeeze(-1)
    Node_T_adj = (m_col_T_adj[:,0].unsqueeze(-1) + g_vt.t() @ K_inv_offset_T_adj)#.squeeze(-1)
    
    Node_disp = torch.cat([u, v], dim=1)  # shape: (N_node_w, 2)
    Node_disp_adj = torch.cat([vd1, vd2], dim=1)  # shape: (N_node_w, 2)

    if model_list[0].MP['design_flag']:
        phases = [m_elem_phase[:, i].unsqueeze(-1) + g_phases[i].t() @ K_inv_offsets_phase[i] for i in range(len(K_inv_offsets_phase))]
        phase_weights = torch.cat(phases, dim=1)
    else:
        phase_weights = m_elem_phase
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    
    # debug phase density
    # index = (elem_x[:,0] >= 160) & (elem_x[:,1] >= 55) & (elem_x[:,1] <= 60)
    # X_phase = elem_x[index]
    # density_phase = phase_density[index]
    # weight_phase = phase_weights[index]
    E_hat = (E_tensor * phase_weights ** p).sum(dim = 1) # shape: [n_elem]
    C1_hat = (E_hat / (1 - nu ** 2))[:, None] # [N_elem_w, 1]
    C2_hat = (E_hat / (2 * (1 + nu)))[:, None]
    phase_weights_dummy = torch.zeros_like(phase_weights)
    phase_weights_dummy[:, 1] = 1.0
    kappa_hat = (kappa_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]
    alpha_hat = (alpha_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]

    # calculate elemental temperature
    T_elem = Node_T[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_elem = T_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_elem = T_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]
    T_int_pt = torch.matmul(N, T_elem).squeeze(-1) # [N_elem_w, N_int, 1]

    # calculate stress and strain at each int point
    u_elem = Node_disp[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem = u_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem = u_elem.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps = torch.matmul(B, u_elem).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11, E22, E12 = eps[..., 0], eps[..., 1], eps[..., 2]  # total strain 
    E11_th = alpha_hat * (T_int_pt.squeeze(-1) - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    E22_th = alpha_hat * (T_int_pt.squeeze(-1) - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    E12_th = torch.zeros_like(E12)                  # [N_elem_w, N_int] thermal strain
    E11_th_no_grad = E11_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    E22_th_no_grad = E22_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    E12_th_no_grad = E12_th           # [N_elem_w, N_int] thermal strain no grad
    E11_m = E11 - E11_th_no_grad  # [N_elem_w, N_int]
    E22_m = E22 - E22_th_no_grad  # [N_elem_w, N_int]
    E12_m = E12# - E12_th_no_grad  # [N_elem_w, N_int]
    S11 = C1_hat * (E11_m + nu * E22_m)
    S22 = C1_hat * (nu * E11_m + E22_m)
    S12 = C2_hat * E12_m

    # calculate stress and strain at each int point
    u_elem_adj = Node_disp_adj[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem_adj = u_elem_adj.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem_adj = u_elem_adj.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps_adj = torch.matmul(B, u_elem_adj).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11_adj, E22_adj, E12_adj = eps_adj[..., 0], eps_adj[..., 1], eps_adj[..., 2] # [N_elem_w, N_int]
    S11_adj = C1_hat * (E11_adj + nu * E22_adj) # [N_elem_w, N_int]
    S22_adj = C1_hat * (nu * E11_adj + E22_adj) # [N_elem_w, N_int]
    S12_adj = C2_hat * E12_adj # [N_elem_w, N_int]

    collocation_x = collocation_x.detach().numpy()
    elem_x = elem_x.detach().numpy()
    conn = conn.detach().numpy()
    u = u.detach().numpy()
    v = v.detach().numpy()
    vd1 = vd1.detach().numpy()
    vd2 = vd2.detach().numpy()
    Node_T = Node_T.detach().numpy()
    Node_T_adj = Node_T_adj.detach().numpy()
    S11 = S11.detach().numpy()
    S22 = S22.detach().numpy()
    S12 = S12.detach().numpy()
    S11_adj = S11_adj.detach().numpy()
    S22_adj = S22_adj.detach().numpy()
    S12_adj = S12_adj.detach().numpy()
    E11_m = E11_m.detach().numpy()
    E22_m = E22_m.detach().numpy()
    E12_m = E12_m.detach().numpy()
    E11_th = E11_th.detach().numpy()
    E22_th = E22_th.detach().numpy()
    E12_th = E12_th.numpy()
    rho = phase_density.unsqueeze(-1).detach().numpy()
    Data = {'collocation_x':collocation_x,'elem_x':elem_x,'conn':conn,
            'u':u,'v':v,'vd1':vd1,'vd2':vd2,'Node_T':Node_T,'Node_T_adj':Node_T_adj,
            'S11':S11,'S22':S22,'S12':S12,'S11_adj':S11_adj,'S22_adj':S22_adj,'S12_adj':S12_adj,
            'E11_m':E11_m,'E22_m':E22_m,'E12_m':E12_m,
            'E11_th':E11_th,'E22_th':E22_th,'E12_th':E12_th,
            'E11_adj':E11_adj,'E22_adj':E22_adj,'E12_adj':E12_adj,
            'rho':rho}
    return Data

def eval_model_adjoint_complete(model_list):
    num_phase = model_list[0].MP['num_phase']
    alpha_tensor = model_list[0].MP['alpha'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    kappa_tensor = model_list[0].MP['kappa'][0]
    deltaT = model_list[0].MP['deltaT']
    D = model_list[0].MP['D'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = model_list[0].MP['P'][0]  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    E_tensor = model_list[0].MP['E'][0] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    wt = model_list[0].MP['wt'][0] # e.g., [1.0, 1.0, 1.0, 1.0] a tensor
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    # f_magnitude = model_list[0].f_magnitude
    # f_index = model_list[0].f_index
    conn = model_list[0].conn
    B = model_list[0].B
    B_T = model_list[0].B_T
    N = model_list[0].N
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]
    T_inf = model_list[0].MP['T_inf']
    thickness = model_list[0].MP['domain']['thickness']
    f_out = model_list[0].MP['f_out']
    a_u = 1/f_out
    a_T = model_list[0].MP['a_T']
    
    for model in model_list:
        model.eval()

    m_col_disp = model_list[0].mean_module_NN_All(collocation_x)
    m_col_disp_adj = model_list[1].mean_module_NN_All(collocation_x)
    m_col_T = model_list[2].mean_module_NN_All(collocation_x)
    m_col_T_adj = model_list[3].mean_module_NN_All(collocation_x)
    m_elem_phase = model_list[4].mean_module_NN_All(elem_x)

    # --- Displacement GP ---
    g_u = model_list[0].independent_kernels[0](model_list[0].train_inputs_per_output[0], collocation_x).evaluate()
    g_v = model_list[0].independent_kernels[1](model_list[0].train_inputs_per_output[1], collocation_x).evaluate()

    # --- Adjoint GP ---
    g_vd1 = model_list[1].independent_kernels[0](model_list[1].train_inputs_per_output[0], collocation_x).evaluate()
    g_vd2 = model_list[1].independent_kernels[1](model_list[1].train_inputs_per_output[1], collocation_x).evaluate()

    # --- T GP ---
    g_t = model_list[2].independent_kernels[0](model_list[2].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # --- T adjoint GP ---
    g_vt = model_list[3].independent_kernels[0](model_list[3].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # --- Phase GPs (only if design_flag = True) ---
    if model_list[0].MP['design_flag']:
        g_phases = []
        for i in range(model_list[4].num_output):
            k_i = model_list[4].independent_kernels[i]
            X_i = model_list[4].train_inputs_per_output[i]
            g_phases.append(k_i(X_i, elem_x).evaluate().detach())
    
    # ====================================================
    # Compute Cholesky decompositions and offsets
    # ====================================================
    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            # --- Displacement kernels ---
            model_list[0].chol_decomp = []
            for i in range(model_list[0].num_output):
                K_i = model_list[0].independent_kernels[i](model_list[0].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[0].chol_decomp.append(L_i)

            # --- Displacement adjoint kernels ---
            model_list[1].chol_decomp = []
            for i in range(model_list[1].num_output):
                K_i = model_list[1].independent_kernels[i](model_list[1].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[1].chol_decomp.append(L_i)
            
            # --- T kernels ---
            model_list[2].chol_decomp = []
            for i in range(model_list[2].num_output):
                K_i = model_list[2].independent_kernels[i](model_list[2].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[2].chol_decomp.append(L_i)
                
            # --- T kernels ---
            model_list[3].chol_decomp = []
            for i in range(model_list[3].num_output):
                K_i = model_list[3].independent_kernels[i](model_list[3].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[3].chol_decomp.append(L_i)

            # --- Phase kernels (only if needed) ---
            if model_list[0].MP['design_flag']:
                model_list[4].chol_decomp = []
                for i in range(model_list[4].num_output):
                    K_i = model_list[4].independent_kernels[i](model_list[4].train_inputs_per_output[i]).evaluate().detach()
                    L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                    model_list[4].chol_decomp.append(L_i)

    # ====================================================
    # Compute K⁻¹·(y − μ)
    # ====================================================

    # --- Displacement ---
    K_inv_offset_u = torch.cholesky_solve(
        model_list[0].train_target_per_output[0].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[0].chol_decomp[0]
    )

    K_inv_offset_v = torch.cholesky_solve(
        model_list[0].train_target_per_output[1].unsqueeze(-1)
        - model_list[0].mean_module_NN_All(model_list[0].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
        model_list[0].chol_decomp[1]
    )

    # --- Displacement Adjoint ---
    K_inv_offset_vd1 = torch.cholesky_solve(
        model_list[1].train_target_per_output[0].unsqueeze(-1)
        - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[1].chol_decomp[0]
    )

    K_inv_offset_vd2 = torch.cholesky_solve(
        model_list[1].train_target_per_output[1].unsqueeze(-1)
        - model_list[1].mean_module_NN_All(model_list[1].train_inputs_per_output[1])[:, 1].unsqueeze(-1),
        model_list[1].chol_decomp[1]
    )

    # --- Temperature ---
    K_inv_offset_T = torch.cholesky_solve(
        model_list[2].train_target_per_output[0].unsqueeze(-1)
        - model_list[2].mean_module_NN_All(model_list[2].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[2].chol_decomp[0]
    )
    # --- Temperature Adjoint---
    K_inv_offset_T_adj = torch.cholesky_solve(
        model_list[3].train_target_per_output[0].unsqueeze(-1)
        - model_list[3].mean_module_NN_All(model_list[3].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[3].chol_decomp[0]
    )

    # --- Phase (if design_flag = True) ---
    if model_list[0].MP['design_flag']:
        K_inv_offsets_phase = []
        for i in range(model_list[4].num_output):
            y_i = model_list[4].train_target_per_output[i]
            X_i = model_list[4].train_inputs_per_output[i]
            mu_i = model_list[4].mean_module_NN_All(X_i)[:, i].unsqueeze(-1)
            offset_i = y_i.unsqueeze(-1) - mu_i
            K_inv_offsets_phase.append(
                torch.cholesky_solve(offset_i, model_list[4].chol_decomp[i])
            )

    u = (m_col_disp[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u)#.squeeze(-1)
    v = (m_col_disp[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v)#.squeeze(-1)
    vd1 = (m_col_disp_adj[:,0].unsqueeze(-1) + g_vd1.t() @ K_inv_offset_vd1)#.squeeze(-1)
    vd2 = (m_col_disp_adj[:,1].unsqueeze(-1) + g_vd2.t() @ K_inv_offset_vd2)#.squeeze(-1)
    Node_T = (m_col_T[:,0].unsqueeze(-1) + g_t.t() @ K_inv_offset_T)#.squeeze(-1)
    Node_T_adj = (m_col_T_adj[:,0].unsqueeze(-1) + g_vt.t() @ K_inv_offset_T_adj)#.squeeze(-1)
    
    Node_disp = torch.cat([u, v], dim=1)  # shape: (N_node_w, 2)
    Node_disp_adj = torch.cat([vd1, vd2], dim=1)  # shape: (N_node_w, 2)

    if model_list[0].MP['design_flag']:
        phases = [m_elem_phase[:, i].unsqueeze(-1) + g_phases[i].t() @ K_inv_offsets_phase[i] for i in range(len(K_inv_offsets_phase))]
        phase_weights = torch.cat(phases, dim=1)
    else:
        phase_weights = m_elem_phase
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    
    # debug phase density
    # index = (elem_x[:,0] >= 160) & (elem_x[:,1] >= 55) & (elem_x[:,1] <= 60)
    # X_phase = elem_x[index]
    # density_phase = phase_density[index]
    # weight_phase = phase_weights[index]
    E_hat = (E_tensor * phase_weights ** p).sum(dim = 1) # shape: [n_elem]
    C1_hat = (E_hat / (1 - nu ** 2))[:, None] # [N_elem_w, 1]
    C2_hat = (E_hat / (2 * (1 + nu)))[:, None]
    # kappa_hat = (kappa_tensor * phase_weights ** p).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]
    phase_weights_dummy = torch.zeros_like(phase_weights)
    phase_weights_dummy[:, 1] = 1.0
    kappa_hat = (kappa_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]
    alpha_hat = (alpha_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]

    # calculate material constants (no gradient for density) for DEM terms
    kappa_no_grad = kappa_hat.detach() # shape: [n_elem,1]
    alpha_no_grad = alpha_hat.detach() # shape: [n_elem,1]
    C1_no_grad = C1_hat.detach() # [N_elem_w, 1]
    C2_no_grad = C2_hat.detach()

    # calculate elemental temperature
    T_elem = Node_T[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_elem = T_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_elem = T_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]
    T_int_pt = torch.matmul(N, T_elem).squeeze(-1) # [N_elem_w, N_int, 1]

    # calculate elemental temperature
    T_adj_elem = Node_T_adj[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_adj_elem = T_adj_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_adj_elem = T_adj_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]

    # calculate stress and strain at each int point
    u_elem = Node_disp[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem = u_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem = u_elem.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps = torch.matmul(B, u_elem).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11, E22, E12 = eps[..., 0], eps[..., 1], eps[..., 2]  # total strain 
    E11_th = alpha_hat * (T_int_pt.squeeze(-1) - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    E22_th = alpha_hat * (T_int_pt.squeeze(-1) - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    E12_th = torch.zeros_like(E12)                  # [N_elem_w, N_int] thermal strain
    E11_th_no_grad = E11_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    E22_th_no_grad = E22_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    E12_th_no_grad = E12_th           # [N_elem_w, N_int] thermal strain no grad
    E11_m = E11 - E11_th_no_grad  # [N_elem_w, N_int]
    E22_m = E22 - E22_th_no_grad  # [N_elem_w, N_int]
    E12_m = E12# - E12_th_no_grad  # [N_elem_w, N_int]
    S11 = C1_hat * (E11_m + nu * E22_m)
    S22 = C1_hat * (nu * E11_m + E22_m)
    S12 = C2_hat * E12_m

    # calculate stress and strain at each int point
    u_elem_adj = Node_disp_adj[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem_adj = u_elem_adj.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem_adj = u_elem_adj.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps_adj = torch.matmul(B, u_elem_adj).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11_adj, E22_adj, E12_adj = eps_adj[..., 0], eps_adj[..., 1], eps_adj[..., 2] # [N_elem_w, N_int]
    S11_adj = C1_hat * (E11_adj + nu * E22_adj) # [N_elem_w, N_int]
    S22_adj = C1_hat * (nu * E11_adj + E22_adj) # [N_elem_w, N_int]
    S12_adj = C2_hat * E12_adj # [N_elem_w, N_int]

    # calculate the augmented objective function
    e11_m, e22_m, e12_m = E11_m.detach(), E22_m.detach(), E12_m.detach()
    e11_adj, e22_adj, e12_adj = E11_adj.detach(), E22_adj.detach(), E12_adj.detach()
    s11 = C1_hat * (e11_m + nu * e22_m) # with grad to rho from modulus
    s22 = C1_hat * (nu * e11_m + e22_m) # with grad to rho from modulus
    s12 = C2_hat * e12_m # with grad to rho from modulus

    # --- compliance ---
    comp_ip = (s11 * e11_adj + s22 * e22_adj + 1.0 * s12 * e12_adj)  # [N_elem_w, N_int]
    comp_vector = torch.sum(comp_ip * wt * detJ * thickness, dim=1)  # [N_elem_w, N_int]
    comp = - a_u * comp_vector.sum()
    comp_grad = torch.autograd.grad(comp,phase_weights,grad_outputs=None,retain_graph=False,create_graph=True)[0].squeeze(-1)

    # calculate the conductivity adjoint term:
    dT = torch.matmul(B_T, T_elem).squeeze(-1) # [N_elem_w, N_int, 2]
    dT_adj = torch.matmul(B_T, T_adj_elem).squeeze(-1) # [N_elem_w, N_int, 2]
    dTdvT = (dT * dT_adj).sum(dim=-1)   # [N_elem_w, N_int]
    kdTdvT =  kappa_hat * (dTdvT.detach())  # [N_elem_w, N_int] with grad to rho from kappa
    kdTdvT_int = (kdTdvT * detJ * wt.unsqueeze(0)) * thickness # [N_elem_w, N_int]
    heat = - a_T * kdTdvT_int.sum()
    heat_grad = torch.autograd.grad(heat,phase_weights,grad_outputs=None,retain_graph=False,create_graph=True)[0].squeeze(-1)

    # calculate the  thermal adjoint term:
    s11_adj = S11_adj.detach() # [N_elem_w, N_int]
    s22_adj = S22_adj.detach() # [N_elem_w, N_int]
    thermal_ip = (s11_adj * E11_th + s22_adj * E22_th)  # [N_elem_w, N_int] with grad to rho from alpha
    thermal_vector = torch.sum(thermal_ip * wt * detJ * thickness, dim=1)  #
    thermal = a_u * thermal_vector.sum()
    thermal_grad = torch.autograd.grad(thermal,phase_weights,grad_outputs=None,retain_graph=False,create_graph=True)[0].squeeze(-1)

    collocation_x = collocation_x.detach().numpy()
    elem_x = elem_x.detach().numpy()
    conn = conn.detach().numpy()
    u = u.detach().numpy()
    v = v.detach().numpy()
    vd1 = vd1.detach().numpy()
    vd2 = vd2.detach().numpy()
    Node_T = Node_T.detach().numpy()
    Node_T_adj = Node_T_adj.detach().numpy()
    S11 = S11.detach().numpy()
    S22 = S22.detach().numpy()
    S12 = S12.detach().numpy()
    S11_adj = S11_adj.detach().numpy()
    S22_adj = S22_adj.detach().numpy()
    S12_adj = S12_adj.detach().numpy()
    E11_m = E11_m.detach().numpy()
    E22_m = E22_m.detach().numpy()
    E12_m = E12_m.detach().numpy()
    E11_th = E11_th.detach().numpy()
    E22_th = E22_th.detach().numpy()
    E12_th = E12_th.numpy()
    comp = comp_vector.unsqueeze(-1).detach().numpy()
    comp_grad = comp_grad.detach().numpy()
    heat = kdTdvT_int.detach().numpy()
    heat_grad = heat_grad.detach().numpy()
    thermal = thermal_vector.unsqueeze(-1).detach().numpy()
    thermal_grad = thermal_grad.detach().numpy()
    rho = phase_density.unsqueeze(-1).detach().numpy()
    Data = {'collocation_x':collocation_x,'elem_x':elem_x,'conn':conn,
            'u':u,'v':v,'vd1':vd1,'vd2':vd2,'Node_T':Node_T,'Node_T_adj':Node_T_adj,
            'S11':S11,'S22':S22,'S12':S12,'S11_adj':S11_adj,'S22_adj':S22_adj,'S12_adj':S12_adj,
            'E11_m':E11_m,'E22_m':E22_m,'E12_m':E12_m,
            'E11_th':E11_th,'E22_th':E22_th,'E12_th':E12_th,
            'E11_adj':E11_adj,'E22_adj':E22_adj,'E12_adj':E12_adj,
            'comp':comp,'comp_grad':comp_grad,'heat':heat,
            'heat_grad':heat_grad,'thermal':thermal,'thermal_grad':thermal_grad,
            'rho':rho}
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

def extract_data_adjoint(save_folder, iter, mesh_key, Example):
    # load the MP
    MP = torch.load(save_folder + 'MP.pt', map_location=torch.device('cpu'))
    base_folder = MP['base_folder']
    N_worker = MP['N_worker']
    Diff_type = MP['Diff_type']
    num_phase = MP["num_phase"]

    # load the NN_config
    with open(save_folder + "NN_config_disp.json", "r") as file:
        NN_config_disp = json.load(file)
    with open(save_folder + "NN_config_rho.json", "r") as file:
        NN_config_rho = json.load(file)
    
    # load the training data:
    train_data = torch.load(save_folder + "train_data.pth",map_location=torch.device('cpu'))

    # u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # vd1_X_train = Training['vd1_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # vd1_train = Training['vd1_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # vd2_X_train = Training['vd2_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # vd2_train = Training['vd2_train'].type(tkwargs["dtype"]).requires_grad_(False)
    # phase_train = Training['phase_train']
    model_disp = LMGP(train_data['disp']['X'], train_data['disp']['y'], NN_config_disp,
                    name_output="disp", MP=MP, num_output=2)
    model_disp_adj = LMGP(train_data['disp_adj']['X'], train_data['disp_adj']['y'], NN_config_disp,
                    name_output="adj", MP=MP, num_output=2)
    model_T = LMGP(train_data['T']['X'], train_data['T']['y'], NN_config_disp,
                    name_output="T", MP=MP, num_output=1)
    model_T_adj = LMGP(train_data['T_adj']['X'], train_data['T_adj']['y'], NN_config_disp,
                    name_output="T_adj", MP=MP, num_output=1)
    model_phase = LMGP(train_data['phase']['X'], train_data['phase']['y'], NN_config_rho,
                    name_output="rho", MP=MP, num_output=num_phase)
    model_list = [model_disp, model_disp_adj, model_T, model_T_adj, model_phase]

    checkpoint = torch.load(save_folder + f'Trained_models_{iter}.pth',map_location=torch.device('cpu'))
    model_list[0].mean_module_NN_All.load_state_dict(checkpoint['model_u_state_dict'])
    model_list[1].mean_module_NN_All.load_state_dict(checkpoint['model_u_adj_state_dict'])
    model_list[2].mean_module_NN_All.load_state_dict(checkpoint['model_T_state_dict'])
    model_list[3].mean_module_NN_All.load_state_dict(checkpoint['model_T_adj_state_dict'])
    model_list[4].mean_module_NN_All.load_state_dict(checkpoint['model_rho_state_dict'])
    
    # load mesh file:
    # file_loc = base_folder + 'Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    file_loc = 'Data/' + f'{Example}_GPU{1}_{Diff_type}_thermomechanical.pt'
    mesh_data = torch.load(file_loc)

    # Save all data in a dictionary
    # assign mesh data to model_u
    model_list[0].collocation_x    = mesh_data['GPU0'][mesh_key]["X_node"]
    model_list[0].elem_x           = mesh_data['GPU0'][mesh_key]["X_elem"]
    model_list[0].elem_vol         = mesh_data['GPU0'][mesh_key]["elem_vol"]
    # model_list[0].f_index          = mesh_data['GPU0'][mesh_key]["f_index"]
    # model_list[0].f_magnitude      = mesh_data['GPU0'][mesh_key]["f_magnitude"]
    # model_list[0].K_in_index       = mesh_data['GPU0'][mesh_key]["K_in_index"]
    # model_list[0].K_in_magnitude   = mesh_data['GPU0'][mesh_key]["K_in_magnitude"]
    model_list[0].K_out_index      = mesh_data['GPU0'][mesh_key]["K_out_index"]
    model_list[0].K_out_magnitude  = mesh_data['GPU0'][mesh_key]["K_out_magnitude"]
    model_list[0].f_adj_index       = mesh_data['GPU0'][mesh_key]["f_adj_index"]
    model_list[0].f_adj_magnitude   = mesh_data['GPU0'][mesh_key]["f_adj_magnitude"]
    model_list[0].conn             = mesh_data['GPU0'][mesh_key]["conn"]
    model_list[0].B                = mesh_data['GPU0'][mesh_key]["B"]
    model_list[0].B_T             = mesh_data['GPU0'][mesh_key]["B_T"]
    model_list[0].N             = mesh_data['GPU0'][mesh_key]["N"]
    model_list[0].detJ             = mesh_data['GPU0'][mesh_key]["detJ"]

    data = eval_model_adjoint_complete(model_list)
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

save_folder = 'Results/EX2D1/AlCuFe_s_20K_1/'
run_list = [1,2,3,4,5,6,7,8,9,10]
Example = 'EX2D1'
iter = [19999]
mesh_key = 'Mesh_03' # one million elements for mesh_01, and 100K for 04
cmap_selected = cmap_phases_AlCuFe # cmap_phases, cmap_phases_AlCuFe, cmap_density cmap_phases_TiCuFe
for run_num in run_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')

    data_list = []
    for _,  i in enumerate(iter):
        # Extract data for each iteration
        data = extract_data_adjoint(run_folder, i, mesh_key, Example)
        collocation_x = data['collocation_x']
        elem_x = data['elem_x']
        conn = data['conn']
        u = data['u']
        v = data['v']
        vd1 = data['vd1']
        vd2 = data['vd2']
        Node_T = data['Node_T']
        Node_T_adj = data['Node_T_adj']
        S11 = data['S11']
        S22 = data['S22']
        S12 = data['S12']
        S11_adj = data['S11_adj']
        S22_adj = data['S22_adj']
        S12_adj = data['S12_adj']
        E11_th = data['E11_th']
        E22_th = data['E22_th']
        E12_th = data['E12_th']
        E11_m = data['E11_m']
        E22_m = data['E22_m']
        E12_m = data['E12_m']
        E11_adj = data['E11_adj']
        E22_adj = data['E22_adj']
        E12_adj = data['E12_adj']
        comp = data['comp']
        comp_grad = data['comp_grad']
        heat = data['heat']
        heat_grad = data['heat_grad']
        thermal = data['thermal']
        thermal_grad = data['thermal_grad']
        rho = data['rho']

        data_list.append({
            'collocation_x': collocation_x,
            'conn': conn,
            'elem_x': elem_x,
            'u': u,
            'v': v,
            'vd1': vd1,
            'vd2': vd2,
            'Node_T': Node_T,
            'Node_T_adj': Node_T_adj,
            'S11_adj':S11_adj,
            'S22_adj':S22_adj,
            'S12_adj':S12_adj,
            'S11':S11,
            'S22':S22,
            'S12':S12,
            'E11_m':E11_m,
            'E22_m':E22_m,
            'E12_m':E12_m,
            'E11_th':E11_th,
            'E22_th':E22_th,
            'E12_th':E12_th,
            'E11_adj':E11_adj,
            'E22_adj':E22_adj,
            'E12_adj':E12_adj,
            'rho':rho,
            'comp':comp,  
            'comp_grad':comp_grad,  
            'heat':heat,  
            'heat_grad':heat_grad,  
            'thermal':thermal,  
            'thermal_grad':thermal_grad,  
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

    # plot displacements
    for i in range(len(iter)):
        data = data_list[i]
        X_node = data['collocation_x']
        X_elem = data['elem_x']
        conn = data['conn']
        u = data['u']
        v = data['v']
        vd1 = data['vd1']
        vd2 = data['vd2']
        Node_T = data['Node_T']
        Node_T_adj = data['Node_T_adj']
        rho = data['rho']
        S11_adj = data['S11_adj']
        S22_adj = data['S22_adj']
        S12_adj = data['S12_adj']
        S11 = data['S11']
        S22 = data['S22']
        S12 = data['S12']
        E11_m = data['E11_m']
        E22_m = data['E22_m']
        E12_m = data['E12_m']
        E11_th = data['E11_th']
        E22_th = data['E22_th']
        E12_th = data['E12_th']
        E11_adj = data['E11_adj']
        E22_adj = data['E22_adj']
        E12_adj = data['E12_adj']
        comp = data['comp']
        comp_grad = data['comp_grad']
        heat = data['heat']
        heat_grad = data['heat_grad']
        thermal = data['thermal']
        thermal_grad = data['thermal_grad']
        
        # # fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # triangles = np.vstack([conn[:, [0, 1, 2]],conn[:, [0, 2, 3]]])
        # triang = mtri.Triangulation(X_node[:, 0], X_node[:, 1], triangles)
        # # plot displacement fields
        # fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        # # --- u displacement ---
        # cf1 = axs[0,0].tricontourf(triang, u.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf1, ax=axs[0,0], shrink=colorbar_shrink)
        # cbar.set_label(r"$u$")
        # axs[0,0].set_xlabel(r"$X$ (mm)")
        # axs[0,0].set_ylabel(r"$Y$ (mm)")
        # axs[0,0].set_title(r"$u$ ")
        # axs[0,0].set_aspect("equal")
        # # --- v displacement ---
        # cf2 = axs[0,1].tricontourf(triang, v.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf2, ax=axs[0,1], shrink=colorbar_shrink)
        # cbar.set_label(r"$v$")
        # axs[0,1].set_xlabel(r"$X$ (mm)")
        # axs[0,1].set_ylabel(r"$Y$ (mm)")
        # axs[0,1].set_title(r"$v$ ")
        # axs[0,1].set_aspect("equal")
        # # --- u_adj displacement ---
        # cf3 = axs[1,0].tricontourf(triang, vd1.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf3, ax=axs[1,0], shrink=colorbar_shrink)
        # cbar.set_label(r"$u_{adj}$")
        # axs[1,0].set_xlabel(r"$X$ (mm)")
        # axs[1,0].set_ylabel(r"$Y$ (mm)")
        # axs[1,0].set_title(r"$u_{adj}$ ")
        # axs[1,0].set_aspect("equal")
        # # --- v_adj displacement ---
        # cf4 = axs[1,1].tricontourf(triang, vd2.flatten(), levels=50, cmap=cmap_contours)
        # char = plt.colorbar(cf4, ax=axs[1,1], shrink=colorbar_shrink)
        # char.set_label(r"$v_{adj}$")
        # axs[1,1].set_xlabel(r"$X$ (mm)")
        # axs[1,1].set_ylabel(r"$Y$ (mm)")
        # axs[1,1].set_title(r"$v_{adj}$ ")
        # axs[1,1].set_aspect("equal")
        # file_name = f"displacement_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # # plt.show()
        # plt.savefig(file_path, format="pdf", dpi=600, bbox_inches="tight")

        # # plot temperature fields
        # fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # # --- u displacement ---
        # cf1 = axs[0].tricontourf(triang, Node_T.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf1, ax=axs[0], shrink=colorbar_shrink)
        # cbar.set_label(r"$T$")
        # axs[0].set_xlabel(r"$X$ (mm)")
        # axs[0].set_ylabel(r"$Y$ (mm)")
        # axs[0].set_title(r"$T$ ")
        # axs[0].set_aspect("equal")
        # # --- v displacement ---
        # cf2 = axs[1].tricontourf(triang, Node_T_adj.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf2, ax=axs[1], shrink=colorbar_shrink)
        # cbar.set_label(r"$T_{adj}$")
        # axs[1].set_xlabel(r"$X$ (mm)")
        # axs[1].set_ylabel(r"$Y$ (mm)")
        # axs[1].set_title(r"$T_{adj}$ ")
        # axs[1].set_aspect("equal")

        # file_name = f"temperature_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # # plt.show()
        # plt.savefig(file_path, format="pdf", dpi=600, bbox_inches="tight")

        # # plot stress and strain
        # # --- Stress plots (S11, S22, S12) ---
        # fig_stress, axs_stress = plt.subplots(2, 3, figsize=(18, 12))
        # plot_field(axs_stress[0,0], conn, X_node, S11, r"$S_{11}$", cmap_contours)
        # plot_field(axs_stress[0,1], conn, X_node, S22, r"$S_{22}$", cmap_contours)
        # plot_field(axs_stress[0,2], conn, X_node, S12, r"$S_{12}$", cmap_contours)
        # # fig_stress.suptitle(f"stress epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_stress[1,0], conn, X_node, S11_adj, r"$S_{11} adj$", cmap_contours)
        # plot_field(axs_stress[1,1], conn, X_node, S22_adj, r"$S_{22} adj$", cmap_contours)
        # plot_field(axs_stress[1,2], conn, X_node, S12_adj, r"$S_{12} adj$", cmap_contours)
        # # fig_stress.suptitle(f"adjoint stress epoch: {iter[i]}", fontsize=18)
        # file_name = f"stress_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # # --- Strain plots (E11, E22, E12) ---
        # fig_strain, axs_strain = plt.subplots(3, 3, figsize=(18, 18))
        # plot_field(axs_strain[0,0], conn, X_node, E11_th, r"$E11_{th}$", cmap_contours)
        # plot_field(axs_strain[0,1], conn, X_node, E22_th, r"$E22_{th}$", cmap_contours)
        # plot_field(axs_strain[0,2], conn, X_node, E12_th, r"$E12_{th}$", cmap_contours)
        # # fig_strain.suptitle(f"thermal strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[1,0], conn, X_node, E11_m, r"$E11_{m}$", cmap_contours)
        # plot_field(axs_strain[1,1], conn, X_node, E22_m, r"$E22_{m}$", cmap_contours)
        # plot_field(axs_strain[1,2], conn, X_node, E12_m, r"$E12_{m}$", cmap_contours)
        # # fig_strain.suptitle(f"mechanical strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[2,0], conn, X_node, E11_adj, r"$E11_{adj}$", cmap_contours)
        # plot_field(axs_strain[2,1], conn, X_node, E22_adj, r"$E22_{adj}$", cmap_contours)
        # plot_field(axs_strain[2,2], conn, X_node, E12_adj, r"$E12_{adj}$", cmap_contours)
        # # fig_strain.suptitle(f"adjoint strain epoch: {iter[i]}", fontsize=18)

        # file_name = f"strain_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # # --- gradient plots ---
        # fig_strain, axs_strain = plt.subplots(3, 3, figsize=(18, 18))
        # plot_field(axs_strain[0,0], conn, X_node, comp, r"$comp$", cmap_contours)
        # plot_field(axs_strain[0,1], conn, X_node, comp_grad[:,0][:,None], r"$comp_0$", cmap_contours)
        # plot_field(axs_strain[0,2], conn, X_node, comp_grad[:,1][:,None], r"$comp_1$", cmap_contours)
        # # fig_strain.suptitle(f"thermal strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[1,0], conn, X_node, heat, r"$heat$", cmap_contours)
        # plot_field(axs_strain[1,1], conn, X_node, heat_grad[:,0][:,None], r"$heat_0$", cmap_contours)
        # plot_field(axs_strain[1,2], conn, X_node, heat_grad[:,0][:,None], r"$heat_1$", cmap_contours)
        # # fig_strain.suptitle(f"mechanical strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[2,0], conn, X_node, thermal, r"$thermal$", cmap_contours)
        # plot_field(axs_strain[2,1], conn, X_node, thermal_grad[:,0][:,None], r"$thermal_0$", cmap_contours)
        # plot_field(axs_strain[2,2], conn, X_node, thermal_grad[:,0][:,None], r"$thermal_1$", cmap_contours)
        # # fig_strain.suptitle(f"adjoint strain epoch: {iter[i]}", fontsize=18)

        # file_name = f"gradient_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # plot density
        rho = np.clip(rho, 0.0, 1.0)
        fig_stress, axs_stress = plt.subplots(1, 1, figsize=(8, 6))
        plot_field(axs_stress, conn, X_node, rho, r"$\rho$", cmap_selected)
        fig_stress.suptitle(f"Density Epoch{iter[i]}", fontsize=18)
        file_name = f"rho_epoch{iter[i]}_pred.pdf"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='pdf', dpi=600)

save_folder = 'Results/EX2D1/Ni_s_20K_1/'
run_list = [1,2,3,4,5,6,7,8,9,10]
Example = 'EX2D1'
iter = [19999]
mesh_key = 'Mesh_03' # one million elements for mesh_01, and 100K for 04
cmap_selected = cmap_density # cmap_phases, cmap_phases_AlCuFe, cmap_density cmap_phases_TiCuFe
for run_num in run_list:
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')

    data_list = []
    for _,  i in enumerate(iter):
        # Extract data for each iteration
        data = extract_data_adjoint(run_folder, i, mesh_key, Example)
        collocation_x = data['collocation_x']
        elem_x = data['elem_x']
        conn = data['conn']
        u = data['u']
        v = data['v']
        vd1 = data['vd1']
        vd2 = data['vd2']
        Node_T = data['Node_T']
        Node_T_adj = data['Node_T_adj']
        S11 = data['S11']
        S22 = data['S22']
        S12 = data['S12']
        S11_adj = data['S11_adj']
        S22_adj = data['S22_adj']
        S12_adj = data['S12_adj']
        E11_th = data['E11_th']
        E22_th = data['E22_th']
        E12_th = data['E12_th']
        E11_m = data['E11_m']
        E22_m = data['E22_m']
        E12_m = data['E12_m']
        E11_adj = data['E11_adj']
        E22_adj = data['E22_adj']
        E12_adj = data['E12_adj']
        comp = data['comp']
        comp_grad = data['comp_grad']
        heat = data['heat']
        heat_grad = data['heat_grad']
        thermal = data['thermal']
        thermal_grad = data['thermal_grad']
        rho = data['rho']

        data_list.append({
            'collocation_x': collocation_x,
            'conn': conn,
            'elem_x': elem_x,
            'u': u,
            'v': v,
            'vd1': vd1,
            'vd2': vd2,
            'Node_T': Node_T,
            'Node_T_adj': Node_T_adj,
            'S11_adj':S11_adj,
            'S22_adj':S22_adj,
            'S12_adj':S12_adj,
            'S11':S11,
            'S22':S22,
            'S12':S12,
            'E11_m':E11_m,
            'E22_m':E22_m,
            'E12_m':E12_m,
            'E11_th':E11_th,
            'E22_th':E22_th,
            'E12_th':E12_th,
            'E11_adj':E11_adj,
            'E22_adj':E22_adj,
            'E12_adj':E12_adj,
            'rho':rho,
            'comp':comp,  
            'comp_grad':comp_grad,  
            'heat':heat,  
            'heat_grad':heat_grad,  
            'thermal':thermal,  
            'thermal_grad':thermal_grad,  
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

    # plot displacements
    for i in range(len(iter)):
        data = data_list[i]
        X_node = data['collocation_x']
        X_elem = data['elem_x']
        conn = data['conn']
        u = data['u']
        v = data['v']
        vd1 = data['vd1']
        vd2 = data['vd2']
        Node_T = data['Node_T']
        Node_T_adj = data['Node_T_adj']
        rho = data['rho']
        S11_adj = data['S11_adj']
        S22_adj = data['S22_adj']
        S12_adj = data['S12_adj']
        S11 = data['S11']
        S22 = data['S22']
        S12 = data['S12']
        E11_m = data['E11_m']
        E22_m = data['E22_m']
        E12_m = data['E12_m']
        E11_th = data['E11_th']
        E22_th = data['E22_th']
        E12_th = data['E12_th']
        E11_adj = data['E11_adj']
        E22_adj = data['E22_adj']
        E12_adj = data['E12_adj']
        comp = data['comp']
        comp_grad = data['comp_grad']
        heat = data['heat']
        heat_grad = data['heat_grad']
        thermal = data['thermal']
        thermal_grad = data['thermal_grad']
        
        # # fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # triangles = np.vstack([conn[:, [0, 1, 2]],conn[:, [0, 2, 3]]])
        # triang = mtri.Triangulation(X_node[:, 0], X_node[:, 1], triangles)
        # # plot displacement fields
        # fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        # # --- u displacement ---
        # cf1 = axs[0,0].tricontourf(triang, u.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf1, ax=axs[0,0], shrink=colorbar_shrink)
        # cbar.set_label(r"$u$")
        # axs[0,0].set_xlabel(r"$X$ (mm)")
        # axs[0,0].set_ylabel(r"$Y$ (mm)")
        # axs[0,0].set_title(r"$u$ ")
        # axs[0,0].set_aspect("equal")
        # # --- v displacement ---
        # cf2 = axs[0,1].tricontourf(triang, v.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf2, ax=axs[0,1], shrink=colorbar_shrink)
        # cbar.set_label(r"$v$")
        # axs[0,1].set_xlabel(r"$X$ (mm)")
        # axs[0,1].set_ylabel(r"$Y$ (mm)")
        # axs[0,1].set_title(r"$v$ ")
        # axs[0,1].set_aspect("equal")
        # # --- u_adj displacement ---
        # cf3 = axs[1,0].tricontourf(triang, vd1.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf3, ax=axs[1,0], shrink=colorbar_shrink)
        # cbar.set_label(r"$u_{adj}$")
        # axs[1,0].set_xlabel(r"$X$ (mm)")
        # axs[1,0].set_ylabel(r"$Y$ (mm)")
        # axs[1,0].set_title(r"$u_{adj}$ ")
        # axs[1,0].set_aspect("equal")
        # # --- v_adj displacement ---
        # cf4 = axs[1,1].tricontourf(triang, vd2.flatten(), levels=50, cmap=cmap_contours)
        # char = plt.colorbar(cf4, ax=axs[1,1], shrink=colorbar_shrink)
        # char.set_label(r"$v_{adj}$")
        # axs[1,1].set_xlabel(r"$X$ (mm)")
        # axs[1,1].set_ylabel(r"$Y$ (mm)")
        # axs[1,1].set_title(r"$v_{adj}$ ")
        # axs[1,1].set_aspect("equal")
        # file_name = f"displacement_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # # plt.show()
        # plt.savefig(file_path, format="pdf", dpi=600, bbox_inches="tight")

        # # plot temperature fields
        # fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # # --- u displacement ---
        # cf1 = axs[0].tricontourf(triang, Node_T.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf1, ax=axs[0], shrink=colorbar_shrink)
        # cbar.set_label(r"$T$")
        # axs[0].set_xlabel(r"$X$ (mm)")
        # axs[0].set_ylabel(r"$Y$ (mm)")
        # axs[0].set_title(r"$T$ ")
        # axs[0].set_aspect("equal")
        # # --- v displacement ---
        # cf2 = axs[1].tricontourf(triang, Node_T_adj.flatten(), levels=50, cmap=cmap_contours)
        # cbar = plt.colorbar(cf2, ax=axs[1], shrink=colorbar_shrink)
        # cbar.set_label(r"$T_{adj}$")
        # axs[1].set_xlabel(r"$X$ (mm)")
        # axs[1].set_ylabel(r"$Y$ (mm)")
        # axs[1].set_title(r"$T_{adj}$ ")
        # axs[1].set_aspect("equal")

        # file_name = f"temperature_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # # plt.show()
        # plt.savefig(file_path, format="pdf", dpi=600, bbox_inches="tight")

        # # plot stress and strain
        # # --- Stress plots (S11, S22, S12) ---
        # fig_stress, axs_stress = plt.subplots(2, 3, figsize=(18, 12))
        # plot_field(axs_stress[0,0], conn, X_node, S11, r"$S_{11}$", cmap_contours)
        # plot_field(axs_stress[0,1], conn, X_node, S22, r"$S_{22}$", cmap_contours)
        # plot_field(axs_stress[0,2], conn, X_node, S12, r"$S_{12}$", cmap_contours)
        # # fig_stress.suptitle(f"stress epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_stress[1,0], conn, X_node, S11_adj, r"$S_{11} adj$", cmap_contours)
        # plot_field(axs_stress[1,1], conn, X_node, S22_adj, r"$S_{22} adj$", cmap_contours)
        # plot_field(axs_stress[1,2], conn, X_node, S12_adj, r"$S_{12} adj$", cmap_contours)
        # # fig_stress.suptitle(f"adjoint stress epoch: {iter[i]}", fontsize=18)
        # file_name = f"stress_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # # --- Strain plots (E11, E22, E12) ---
        # fig_strain, axs_strain = plt.subplots(3, 3, figsize=(18, 18))
        # plot_field(axs_strain[0,0], conn, X_node, E11_th, r"$E11_{th}$", cmap_contours)
        # plot_field(axs_strain[0,1], conn, X_node, E22_th, r"$E22_{th}$", cmap_contours)
        # plot_field(axs_strain[0,2], conn, X_node, E12_th, r"$E12_{th}$", cmap_contours)
        # # fig_strain.suptitle(f"thermal strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[1,0], conn, X_node, E11_m, r"$E11_{m}$", cmap_contours)
        # plot_field(axs_strain[1,1], conn, X_node, E22_m, r"$E22_{m}$", cmap_contours)
        # plot_field(axs_strain[1,2], conn, X_node, E12_m, r"$E12_{m}$", cmap_contours)
        # # fig_strain.suptitle(f"mechanical strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[2,0], conn, X_node, E11_adj, r"$E11_{adj}$", cmap_contours)
        # plot_field(axs_strain[2,1], conn, X_node, E22_adj, r"$E22_{adj}$", cmap_contours)
        # plot_field(axs_strain[2,2], conn, X_node, E12_adj, r"$E12_{adj}$", cmap_contours)
        # # fig_strain.suptitle(f"adjoint strain epoch: {iter[i]}", fontsize=18)

        # file_name = f"strain_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # # --- gradient plots ---
        # fig_strain, axs_strain = plt.subplots(3, 3, figsize=(18, 18))
        # plot_field(axs_strain[0,0], conn, X_node, comp, r"$comp$", cmap_contours)
        # plot_field(axs_strain[0,1], conn, X_node, comp_grad[:,0][:,None], r"$comp_0$", cmap_contours)
        # plot_field(axs_strain[0,2], conn, X_node, comp_grad[:,1][:,None], r"$comp_1$", cmap_contours)
        # # fig_strain.suptitle(f"thermal strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[1,0], conn, X_node, heat, r"$heat$", cmap_contours)
        # plot_field(axs_strain[1,1], conn, X_node, heat_grad[:,0][:,None], r"$heat_0$", cmap_contours)
        # plot_field(axs_strain[1,2], conn, X_node, heat_grad[:,0][:,None], r"$heat_1$", cmap_contours)
        # # fig_strain.suptitle(f"mechanical strain epoch: {iter[i]}", fontsize=18)
        # plot_field(axs_strain[2,0], conn, X_node, thermal, r"$thermal$", cmap_contours)
        # plot_field(axs_strain[2,1], conn, X_node, thermal_grad[:,0][:,None], r"$thermal_0$", cmap_contours)
        # plot_field(axs_strain[2,2], conn, X_node, thermal_grad[:,0][:,None], r"$thermal_1$", cmap_contours)
        # # fig_strain.suptitle(f"adjoint strain epoch: {iter[i]}", fontsize=18)

        # file_name = f"gradient_epoch{iter[i]}.pdf"
        # file_path = f"{run_folder}/{file_name}"
        # plt.savefig(file_path, format='pdf', dpi=600)

        # plot density
        rho = np.clip(rho, 0.0, 1.0)
        fig_stress, axs_stress = plt.subplots(1, 1, figsize=(8, 6))
        plot_field(axs_stress, conn, X_node, rho, r"$\rho$", cmap_selected)
        fig_stress.suptitle(f"Density Epoch{iter[i]}", fontsize=18)
        file_name = f"rho_epoch{iter[i]}_pred.pdf"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='pdf', dpi=600)



