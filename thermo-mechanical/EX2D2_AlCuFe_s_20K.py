import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import numpy as np
import time
import shutil
from models.lmgp_updated3 import LMGP 
from gpytorch.settings import cholesky_jitter
from tqdm import tqdm
from utils.utils_general import set_seed0, get_multiGPU
from utils.get_training_data_2D import get_data_EX2D2_thermomech_actuator_four_phase_source as get_data

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def calculate_T_mean(model_list):
    collocation_x = model_list[0].collocation_x.clone()

    for model in model_list:
        model.train()

    m_col_T = model_list[2].mean_module_NN_All(collocation_x)

     # --- T adjoint GP ---
    g_t = model_list[2].independent_kernels[0](model_list[2].train_inputs_per_output[0], collocation_x).evaluate().detach()

    # ====================================================
    # Compute Cholesky decompositions and offsets
    # ====================================================
    if model_list[2].chol_decomp is None:
        with cholesky_jitter(1e-5):
            # --- T kernels ---
            model_list[2].chol_decomp = []
            for i in range(model_list[2].num_output):
                K_i = model_list[2].independent_kernels[i](model_list[2].train_inputs_per_output[i]).evaluate().detach()
                L_i = torch.linalg.cholesky(K_i + 1e-5 * torch.eye(K_i.shape[0], device=K_i.device))
                model_list[2].chol_decomp.append(L_i)
    # --- Temperature ---
    K_inv_offset_T = torch.cholesky_solve(
        model_list[2].train_target_per_output[0].unsqueeze(-1)
        - model_list[2].mean_module_NN_All(model_list[2].train_inputs_per_output[0])[:, 0].unsqueeze(-1),
        model_list[2].chol_decomp[0]
    )
    
    Node_T = (m_col_T[:,0].unsqueeze(-1) + g_t.t() @ K_inv_offset_T)#.squeeze(-1)

    # calculate the mean temperature
    T_mean = Node_T.mean()

    return T_mean

def calculate_TO_loss(model_list):
    MP = model_list[0].MP
    num_phase = MP['num_phase']
    alpha_tensor = MP['alpha'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    kappa_tensor = MP['kappa'][0]
    D = MP['D'][0]  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = MP['P'][0]  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    E_tensor = MP['E'][0] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    wt = MP['wt'][0] # e.g., [1.0, 1.0, 1.0, 1.0] a tensor
    f_out = MP['f_out']
    a_u = 1/f_out
    a_T = MP['a_T']
    K_out = MP['K_out']
    # deltaT = MP['deltaT']
    nu = MP['nu']
    p = MP['p']
    s_tensor = model_list[0].MP['s'][0]
    T_inf = model_list[0].MP['T_inf']
    thickness = MP['domain']['thickness']

    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()
    elem_vol = model_list[0].elem_vol
    f_adj_index = model_list[0].f_adj_index
    f_adj_magnitude = f_out * model_list[0].f_adj_magnitude
    K_out_index = model_list[0].K_out_index
    K_out_magnitude = K_out * model_list[0].K_out_magnitude
    conn = model_list[0].conn
    B = model_list[0].B
    B_T = model_list[0].B_T
    N = model_list[0].N
    detJ = model_list[0].detJ
    N_elem_w = elem_x.shape[0]

    for model in model_list:
        model.train()

    # ======================
    # 1. Mean predictions
    # ======================
    m_col_disp = model_list[0].mean_module_NN_All(collocation_x)
    m_col_disp_adj = model_list[1].mean_module_NN_All(collocation_x)
    m_col_T = model_list[2].mean_module_NN_All(collocation_x)
    m_col_T_adj = model_list[3].mean_module_NN_All(collocation_x)
    m_elem_phase = model_list[4].mean_module_NN_All(elem_x)

    # ====================================================
    # Independent kernels 
    # ====================================================
    # --- Displacement GP ---
    g_u = model_list[0].independent_kernels[0](model_list[0].train_inputs_per_output[0], collocation_x).evaluate().detach()
    g_v = model_list[0].independent_kernels[1](model_list[0].train_inputs_per_output[1], collocation_x).evaluate().detach()

    # --- Displacement adjoint GP ---
    g_vd1 = model_list[1].independent_kernels[0](model_list[1].train_inputs_per_output[0], collocation_x).evaluate().detach()
    g_vd2 = model_list[1].independent_kernels[1](model_list[1].train_inputs_per_output[1], collocation_x).evaluate().detach()

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
    
    # calculate the mean temperature
    # T_mean = Node_T.mean().detach()
    # calculate the density for each element block
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    mass = (phase_density * elem_vol).sum()
    M0 = (elem_vol).sum()
    # massfrac = mass / M0
    # loss_mConstraint = torch.square(torch.max((massfrac / massfrac_star) - 1.0, 0.0*mass))
    # loss_mConstraint = torch.square(torch.clamp((massfrac / massfrac_star) - 1.0, min=0.0))

    # Compute interpolated density
    phase_cost = (phase_weights * P).sum(dim=1)  # shape: [n_elem]
    cost = (phase_cost * elem_vol).sum()
    cost0 = (P.max()*torch.ones_like(phase_cost) * elem_vol).sum()
    # costfrac = cost / cost0
    # loss_cConstraint = torch.square(torch.max((costfrac / costfrac_star) - 1.0, 0.0*cost))
    # loss_cConstraint = torch.square(torch.clamp((costfrac / costfrac_star) - 1.0, min=0.0))
    
    # Calculate fraction of gray element number per phase 
    grey_counts = []
    weights_local = phase_weights.detach()  # [N_elem_w, num_phase]

    for i in range(num_phase):
        mask = (weights_local[:, i] > MP['rho_min']) & (weights_local[:, i] < MP['rho_max'])
        grey_counts.append(mask.sum())  # scalar tensor per phase

    # Convert list -> single tensor [num_phase]
    grey_counts = torch.stack(grey_counts, dim=0)  # shape: [num_phase]
    count_total = torch.tensor(weights_local.shape[0], device=weights_local.device)

    # calculate material constants (with gradient for density) for the adjoint terms
    
    E_hat = (E_tensor * phase_weights ** p).sum(dim = 1) # shape: [n_elem]
    C1_hat = (E_hat / (1 - nu ** 2))[:, None] # [N_elem_w, 1]
    C2_hat = (E_hat / (2 * (1 + nu)))[:, None]
    kappa_hat = (kappa_tensor * phase_weights ** p).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]
    alpha_hat = (alpha_tensor * phase_weights).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]
    s_hat = (s_tensor * phase_weights ** p).sum(dim = 1).unsqueeze(-1) # shape: [n_elem,1]

    # calculate material constants (no gradient for density) for DEM terms
    kappa_no_grad = kappa_hat.detach() # shape: [n_elem,1]
    alpha_no_grad = alpha_hat.detach() # shape: [n_elem,1]
    s_no_grad = s_hat.detach() # shape: [n_elem,1]
    C1_no_grad = C1_hat.detach() # [N_elem_w, 1]
    C2_no_grad = C2_hat.detach()

    # ========================
    # 1. estimate primal equations for temperature
    # ========================
    # calculate elemental temperature
    T_elem = Node_T[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_elem = T_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_elem = T_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]
    
    # calculate source energy
    T_int_pt = torch.matmul(N, T_elem).squeeze(-1) # [N_elem_w, N_int, 1]
    sT = s_no_grad.unsqueeze(-1) * T_int_pt              # [N_elem_w, N_int, 1]
    sT_int = (sT.squeeze(-1) * detJ * wt.unsqueeze(0)) * thickness   # [N_elem_w, N_int]
    source_energy = sT_int.sum()

    # calculate heat energy
    dT = torch.matmul(B_T, T_elem).squeeze(-1) # [N_elem_w, N_int, 2]
    dT_square = (dT ** 2).sum(dim=-1)   # [N_elem_w, N_int]
    kdT2 =  kappa_no_grad * dT_square  # [N_elem_w, N_int]
    kdT2_int = (kdT2 * detJ * wt.unsqueeze(0)) * thickness # [N_elem_w, N_int]
    heat_energy = 0.5 * kdT2_int.sum()

    # ========================
    # 2. estimate primal equations for displacement
    # ========================
    # calculate the external work
    external_work = 0

    # calculate the stored spring energy
    # disp_K_in = Node_disp[K_in_index]
    # external_work_K_in = 0.5 * torch.sum((disp_K_in**2) * K_in_magnitude)
    disp_K_out = Node_disp[K_out_index]
    external_work_K_out = 0.5 * torch.sum((disp_K_out**2) * K_out_magnitude)
    # spring_energy = external_work_K_in + external_work_K_out    
    spring_energy = external_work_K_out   

    # calculate stress and strain at each int point
    u_elem = Node_disp[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem = u_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem = u_elem.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps = torch.matmul(B, u_elem).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11, E22, E12 = eps[..., 0], eps[..., 1], eps[..., 2]  # total strain 
    E11_th = alpha_hat * (T_int_pt.squeeze(-1).detach() - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    E22_th = alpha_hat * (T_int_pt.squeeze(-1).detach() - T_inf)  # [N_elem_w, N_int] thermal strain with grad, must contain gradient to rho for thermal adjoint term
    #E12_th = torch.zeros_like(E12)                  # [N_elem_w, N_int] thermal strain
    E11_th_no_grad = E11_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    E22_th_no_grad = E22_th.detach()  # [N_elem_w, N_int] thermal strain no grad
    #E12_th_no_grad = E12_th           # [N_elem_w, N_int] thermal strain no grad
    E11_m = E11 - E11_th_no_grad  # [N_elem_w, N_int]
    E22_m = E22 - E22_th_no_grad  # [N_elem_w, N_int]
    E12_m = E12# - E12_th_no_grad  # [N_elem_w, N_int]
    S11 = C1_no_grad * (E11_m + nu * E22_m)
    S22 = C1_no_grad * (nu * E11_m + E22_m)
    S12 = C2_no_grad * E12_m

    # --- Strain energy ---
    SE_ip = 0.5 * (S11 * E11_m + S22 * E22_m + 1.0 * S12 * E12_m)  # [N_elem_w, N_int]
    SE_vector = torch.sum(SE_ip * wt * detJ * thickness, dim=1)  #
    strain_energy = SE_vector.sum()

    # ========================
    # 3. estimate adjoint equations for displacement
    # ========================
    # calculate the external work
    disp_f_adj = Node_disp_adj[f_adj_index]
    external_work_adj = torch.sum(disp_f_adj * f_adj_magnitude)

    # calculate the stored spring energy
    # disp_K_in_adj = Node_test[K_in_index]
    # external_work_K_in_adj = 0.5 * torch.sum((disp_K_in_adj**2) * K_in_magnitude)
    disp_K_out_adj = Node_disp_adj[K_out_index]
    external_work_K_out_adj = 0.5 * torch.sum((disp_K_out_adj**2) * K_out_magnitude)
    spring_energy_adj = external_work_K_out_adj

    # calculate stress and strain at each int point
    u_elem_adj = Node_disp_adj[conn].reshape(N_elem_w, -1) # [N_elem_w, 8] for CSP4
    u_elem_adj = u_elem_adj.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 8, 1]
    u_elem_adj = u_elem_adj.expand(-1, B.shape[1], -1, -1)     # [N_elem_w, N_int, 8, 1]
    eps_adj = torch.matmul(B, u_elem_adj).squeeze(-1)              # [N_elem_w, N_int, 3]
    E11_adj, E22_adj, E12_adj = eps_adj[..., 0], eps_adj[..., 1], eps_adj[..., 2] # [N_elem_w, N_int]
    S11_adj = C1_no_grad * (E11_adj + nu * E22_adj) # [N_elem_w, N_int]
    S22_adj = C1_no_grad * (nu * E11_adj + E22_adj) # [N_elem_w, N_int]
    S12_adj = C2_no_grad * E12_adj # [N_elem_w, N_int]

    # --- Strain energy ---
    SE_ip_adj = 0.5 * (S11_adj * E11_adj + S22_adj * E22_adj + 1.0 * S12_adj * E12_adj)  # [N_elem_w, N_int]
    SE_vector_adj = torch.sum(SE_ip_adj * wt * detJ * thickness, dim=1)  #
    strain_energy_adj = SE_vector_adj.sum()

    # ========================
    # 4. estimate adjoint equations for temperature
    # ========================
    # calculate elemental temperature
    T_adj_elem = Node_T_adj[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_adj_elem = T_adj_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_adj_elem = T_adj_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]
    
    # calculate source
    s11_adj = S11_adj.detach() # [N_elem_w, N_int]
    s22_adj = S22_adj.detach() # [N_elem_w, N_int]
    s_adj = (a_u/a_T) * (alpha_no_grad * (S11_adj + S22_adj)).detach() # s_adj:alpha I: [N_elem_w, N_int]

    # calculate source energy
    T_adj_int_pt = torch.matmul(N, T_adj_elem).squeeze(-1) # [N_elem_w, N_int, 1]
    sT_adj = s_adj.unsqueeze(-1) * T_adj_int_pt              # [N_elem_w, N_int, 1]
    sT_int_adj = (sT_adj.squeeze(-1) * detJ * wt.unsqueeze(0)) * thickness   # [N_elem_w, N_int]
    source_energy_adj = sT_int_adj.sum()

    # calculate heat energy
    dT_adj = torch.matmul(B_T, T_adj_elem).squeeze(-1) # [N_elem_w, N_int, 2]
    dT_adj_square = (dT_adj ** 2).sum(dim=-1)   # [N_elem_w, N_int]
    kdT2_adj =  kappa_no_grad * dT_adj_square  # [N_elem_w, N_int]
    kdT2_adj_int = (kdT2_adj * detJ * wt.unsqueeze(0)) * thickness # [N_elem_w, N_int]
    heat_energy_adj = 0.5 * kdT2_adj_int.sum()

    # ========================
    # 5. calculate the augmented objective function
    # ========================
    # calculate material constants (with gradient for density)
    e11_m, e22_m, e12_m = E11_m.detach(), E22_m.detach(), E12_m.detach()
    e11_adj, e22_adj, e12_adj = E11_adj.detach(), E22_adj.detach(), E12_adj.detach()
    s11 = C1_hat * (e11_m + nu * e22_m) # with grad to rho from modulus
    s22 = C1_hat * (nu * e11_m + e22_m) # with grad to rho from modulus
    s12 = C2_hat * e12_m # with grad to rho from modulus

    # --- compliance ---
    comp_ip = (s11 * e11_adj + s22 * e22_adj + 1.0 * s12 * e12_adj)  # [N_elem_w, N_int]
    comp_vector = torch.sum(comp_ip * wt * detJ * thickness, dim=1)  #
    comp = a_u * comp_vector.sum()

    # calculate the conductivity adjoint term:
    dTdvT = (dT * dT_adj).sum(dim=-1)   # [N_elem_w, N_int]
    kdTdvT =  kappa_hat * (dTdvT.detach())  # [N_elem_w, N_int] with grad to rho from kappa
    kdTdvT_int = (kdTdvT * detJ * wt.unsqueeze(0)) * thickness # [N_elem_w, N_int]
    heat = a_T * kdTdvT_int.sum()

    # calculate the  thermal adjoint term:
    thermal_ip = (s11_adj * E11_th + s22_adj * E22_th)  # [N_elem_w, N_int] with grad to rho from alpha
    thermal_vector = torch.sum(thermal_ip * wt * detJ * thickness, dim=1)  #
    thermal = a_u * thermal_vector.sum()

    # calculate the  source adjoint term:
    source_ip = (s_hat.unsqueeze(-1) * T_adj_int_pt.detach())  # [N_elem_w, N_int, 1] with grad to rho from alpha
    source_vector = torch.sum(source_ip.squeeze(-1) * wt * detJ * thickness, dim=1)  #
    source = a_T * source_vector.sum()

    # --- augmented objective --- 
    obj = - disp_K_out[0,0].detach() - comp - heat + thermal + source
    # ========================
    # Assemble loss dictionary to return
    # ========================
    loss_dict = {
        # --- primal and adjoint objectives ---
        'obj_func': obj,                # augmented objective
        'strain_energy': strain_energy,
        'spring_energy': spring_energy,
        'external_work': external_work,     
        'strain_energy_adj': strain_energy_adj,
        'spring_energy_adj': spring_energy_adj,
        'external_work_adj': external_work_adj,

        # --- thermal terms ---
        'heat_energy': heat_energy,
        'source_energy': source_energy,
        'heat_energy_adj': heat_energy_adj,
        'source_energy_adj': source_energy_adj,

        # --- material/volume metrics ---
        'mass': mass,
        'M0': M0,
        'cost': cost,
        'cost0': cost0,
        'grey_counts': grey_counts,
        'count_total': count_total,
    }

    return loss_dict

# ========================
# Worker: single GPU debug
# ========================
def run_worker(MP, NN_config_disp, NN_config_rho, train_data, index_CP, mesh_data):
    device = torch.device("cuda:0")

    Example = MP["Example"]
    Case = NN_config_disp["Case"]
    base_folder = MP["base_folder"]
    run_folder = f"{base_folder}/Results/{Example}/{Case}/"

    num_phase = MP["num_phase"]
    frac_decrease = MP['frac_decrease']
    TO_num_iter = NN_config_disp["TO_num_iter"]
    plotting_interval_TO = NN_config_disp["plotting_interval_TO"]
    delta = NN_config_disp["delta"]
    wc = NN_config_disp["wc"]
    wd = NN_config_disp["wd"]
    wm = NN_config_disp["wm"]
    wp = NN_config_disp["wp"]
    wtemp = NN_config_disp["wtemp"]
    gradient_clip = NN_config_disp["gradient_clip"]
    nrmThreshold = NN_config_disp["nrmThreshold"]
    dynamic_weight = NN_config_disp["dynamic_weight"]
    random_state = NN_config_disp["random_state"]
    learning_rate_disp = NN_config_disp["learning_rate_disp"]
    learning_rate_disp_adj = NN_config_disp["learning_rate_disp_adj"]
    learning_rate_T = NN_config_disp["learning_rate_T"]
    learning_rate_T_adj = NN_config_disp["learning_rate_T_adj"]
    learning_rate_rho = NN_config_disp["learning_rate_rho"]
    Diff_type = MP["Diff_type"]

    # loop over random states (multiple runs)
    for i, state in enumerate(random_state, start=1):
        set_seed0(state)
        NN_config_disp["state"] = state

        # save folder
        save_folder = f"{run_folder}Run_{i}/"
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        NN_config_disp["save_folder"] = save_folder
        NN_config_rho["save_folder"] = save_folder

        # ========================
        # Build models
        # ========================
        model_disp = LMGP(train_data['disp']['X'], train_data['disp']['y'], NN_config_disp,
                        name_output="disp", MP=MP, num_output=2).to_device(device)
        model_disp_adj = LMGP(train_data['disp_adj']['X'], train_data['disp_adj']['y'], NN_config_disp,
                        name_output="adj", MP=MP, num_output=2).to_device(device)
        model_T = LMGP(train_data['T']['X'], train_data['T']['y'], NN_config_disp,
                        name_output="T", MP=MP, num_output=1).to_device(device)
        model_T_adj = LMGP(train_data['T_adj']['X'], train_data['T_adj']['y'], NN_config_disp,
                        name_output="T_adj", MP=MP, num_output=1).to_device(device)
        model_phase = LMGP(train_data['phase']['X'], train_data['phase']['y'], NN_config_rho,
                        name_output="rho", MP=MP, num_output=num_phase).to_device(device)

        # Keep all in list for BC handling
        model_list = [model_disp, model_disp_adj, model_T, model_T_adj, model_phase]

        # define the time history dict
        timeHistory = {
            'loss_total': [], 'loss_obj': [], 'loss_dem_disp': [],'loss_dem_disp_adj':[],
            'loss_dem_t': [],'loss_dem_t_adj':[],
            'loss_mConstraint': [], 'loss_cConstraint': [], 'loss_tv': [], 
            'strain_energy': [], 'external_work': [], 'spring_energy':[],
            'strain_energy_adj': [], 'external_work_adj': [], 'spring_energy_adj':[],
            'heat_energy': [], 'source_energy': [], 
            'heat_energy_adj': [], 'source_energy_adj': [], 
            'loss_pde_1': [], 'loss_pde_2': [],
            'massfrac': [], 'costfrac': [],
            'wc': [], 'wd': [], 'wm': [], 'wp': [],'wtemp':[],
            'grey': [], 'p': []
        }

        # ========================
        # Optimizers + schedulers
        # ========================
        optimizer_disp = torch.optim.Adam(model_list[0].parameters(),
                                          lr=learning_rate_disp, amsgrad=True)
        optimizer_disp_adj = torch.optim.Adam(model_list[1].parameters(),
                                          lr=learning_rate_disp_adj, amsgrad=True)
        optimizer_T = torch.optim.Adam(model_list[2].parameters(),
                                          lr=learning_rate_T, amsgrad=True)
        optimizer_T_adj = torch.optim.Adam(model_list[3].parameters(),
                                          lr=learning_rate_T_adj, amsgrad=True)
        optimizer_rho = torch.optim.Adam(model_list[4].parameters(),
                                         lr=learning_rate_rho, amsgrad=True)

        scheduler_disp = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disp,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_disp_adj = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disp_adj,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_T = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_T,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_T_adj = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_T_adj,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )
        scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_rho,
            milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
            gamma=0.75
        )

        # ========================
        # Training loop
        # ========================
        epochs_iter = tqdm(range(TO_num_iter),
                           desc="GPU0 TO Epoch",
                           position=0, leave=True)
        total_time, start_time = 0, time.time()

        for epoch in epochs_iter:
            optimizer_disp.zero_grad()
            optimizer_disp_adj.zero_grad()
            optimizer_T.zero_grad()
            optimizer_T_adj.zero_grad()
            optimizer_rho.zero_grad()

            # --- mesh slice ---
            index = index_CP[epoch]
            mesh_key = f"Mesh_{index:02d}"
            GPU_key = "GPU0"  # single GPU

            # assign mesh data to model_u
            model_list[0].collocation_x = mesh_data[GPU_key][mesh_key]["X_node"]
            model_list[0].elem_x        = mesh_data[GPU_key][mesh_key]["X_elem"]
            model_list[0].elem_vol        = mesh_data[GPU_key][mesh_key]["elem_vol"]
            # model_list[0].f_index       = mesh_data[GPU_key][mesh_key]["f_index"]
            # model_list[0].f_magnitude   = mesh_data[GPU_key][mesh_key]["f_magnitude"]
            # model_list[0].K_in_index       = mesh_data[GPU_key][mesh_key]["K_in_index"]
            # model_list[0].K_in_magnitude   = mesh_data[GPU_key][mesh_key]["K_in_magnitude"]
            model_list[0].K_out_index       = mesh_data[GPU_key][mesh_key]["K_out_index"]
            model_list[0].K_out_magnitude   = mesh_data[GPU_key][mesh_key]["K_out_magnitude"]
            model_list[0].f_adj_index       = mesh_data[GPU_key][mesh_key]["f_adj_index"]
            model_list[0].f_adj_magnitude   = mesh_data[GPU_key][mesh_key]["f_adj_magnitude"]
            model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
            model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
            model_list[0].B_T             = mesh_data[GPU_key][mesh_key]["B_T"]
            model_list[0].N             = mesh_data[GPU_key][mesh_key]["N"]
            model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]
            
            # bias shift for the temperature NN to match TD:
            if epoch == 0:
                with torch.no_grad():
                    T_mean = calculate_T_mean(model_list)
                    bias_shift = MP['TD'] - T_mean.item()   # scalar difference
                    # Update only the bias of the last layer of the temperature network
                    model_list[2].mean_module_NN_All.network.last.bias += bias_shift
                    T_mean_updated = calculate_T_mean(model_list)
                    print(f'Initial mean temperature (K): {T_mean_updated}')

            
            # --- calculate local losses ---
            loss_dict = calculate_TO_loss(model_list)
            
            # for multi-GPU, assemble the global quantities here:
            obj_global           = loss_dict['obj_func']
            strain_energy_global = loss_dict['strain_energy']
            spring_energy_global = loss_dict['spring_energy']
            external_work_global = loss_dict['external_work']
            strain_energy_adj_global = loss_dict['strain_energy_adj']
            spring_energy_adj_global = loss_dict['spring_energy_adj']
            external_work_adj_global = loss_dict['external_work_adj']
            heat_energy_global = loss_dict['heat_energy']
            source_energy_global = loss_dict['source_energy']
            heat_energy_adj_global = loss_dict['heat_energy_adj']
            source_energy_adj_global = loss_dict['source_energy_adj']
            mass_global          = loss_dict['mass']
            M0_global            = loss_dict['M0']
            cost_global          = loss_dict['cost']
            cost0_global         = loss_dict['cost0']
            grey_counts_global   = loss_dict['grey_counts']
            count_total_global   = loss_dict['count_total']

            # calculate fraction 
            massfrac = mass_global / M0_global
            costfrac = cost_global / cost0_global
            grey_fraction = grey_counts_global / count_total_global
            
            # set initial volume or cost fraction based on the initial prediction of NN
            if epoch == 0:
                initial_massfrac = massfrac.detach().item()
                initial_costfrac = costfrac.detach().item()
                frac_step_mass = (initial_massfrac - massfrac_f)/frac_decrease
                frac_step_cost = (initial_costfrac - costfrac_f)/frac_decrease
                frac_step_p = (MP['pf'] - MP['p0'])/frac_decrease

                model_list[0].MP['p'] = MP['p0']
                model_list[0].MP['massfrac_star'] = initial_massfrac
                model_list[0].MP['costfrac_star'] = initial_costfrac
                model_list[0].MP['frac_step_p'] = frac_step_p
                model_list[0].MP['frac_step_mass'] = frac_step_mass
                model_list[0].MP['frac_step_cost'] = frac_step_cost
                MP['frac_step_mass'] = frac_step_mass
                MP['frac_step_cost'] = frac_step_cost
                MP['frac_step_p'] = frac_step_p
                MP['massfrac0'] = model_list[0].MP['massfrac_star']
                MP['costfrac0'] = model_list[0].MP['costfrac_star']


            # --- build loss terms ---
            massfrac_star = model_list[0].MP['massfrac_star']
            costfrac_star = model_list[0].MP['costfrac_star']
            loss_obj = obj_global
            # offset_dem = (1 + delta) / 2 * external_work_global.detach()
            loss_dem_disp = strain_energy_global + spring_energy_global - external_work_global# + offset_dem
            offset_disp_adj = (1 + delta) / 2 * external_work_adj_global.detach()
            loss_dem_disp_adj = strain_energy_adj_global + spring_energy_adj_global - external_work_adj_global + offset_disp_adj
            loss_dem_t = heat_energy_global - source_energy_global
            loss_dem_t_adj = heat_energy_adj_global - source_energy_adj_global
            if epoch < frac_decrease:
                loss_mConstraint = torch.square((massfrac / massfrac_star) - 1.0)
                loss_cConstraint = torch.square((costfrac / costfrac_star) - 1.0)
            else:
                loss_mConstraint = torch.square(torch.max((massfrac / massfrac_star) - 1.0, 0.0*mass_global))
                loss_cConstraint = torch.square(torch.max((costfrac / costfrac_star) - 1.0, 0.0*cost_global))
            loss =  wc * loss_obj + wd * (loss_dem_disp + loss_dem_disp_adj) + wtemp * (loss_dem_t + loss_dem_t_adj) + wm * loss_mConstraint #+ wp * loss_cConstraint
            # loss =   wtemp * (loss_dem_t)
            # loss =   wd * (loss_dem_disp) + wtemp * (loss_dem_t)
            
            # --- backward ---
            loss.backward(retain_graph=True)

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[1].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[2].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[3].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[4].parameters(), nrmThreshold)

            optimizer_disp.step()
            optimizer_disp_adj.step()
            optimizer_T.step()
            optimizer_T_adj.step()
            optimizer_rho.step()
            scheduler_disp.step()
            scheduler_disp_adj.step()
            scheduler_T.step()
            scheduler_T_adj.step()
            scheduler_rho.step()

            # save the loss histories
            timeHistory['loss_total'].append(loss.item()) 
            timeHistory['loss_obj'].append(loss_obj.item()) 
            timeHistory['loss_dem_disp'].append(loss_dem_disp.item()) 
            timeHistory['loss_dem_disp_adj'].append(loss_dem_disp_adj.item())
            timeHistory['loss_dem_t'].append(loss_dem_t.item())
            timeHistory['loss_dem_t_adj'].append(loss_dem_t_adj.item())
            timeHistory['loss_mConstraint'].append(loss_mConstraint.item()) 
            timeHistory['loss_cConstraint'].append(loss_cConstraint.item()) 
            timeHistory['massfrac'].append(massfrac.item()) 
            timeHistory['costfrac'].append(costfrac.item()) 
            timeHistory['strain_energy'].append(strain_energy_global.item()) 
            timeHistory['spring_energy'].append(spring_energy_global.item()) 
            timeHistory['heat_energy'].append(heat_energy_global.item()) 
            timeHistory['source_energy'].append(source_energy_global.item())
            # timeHistory['external_work'].append(external_work.item())
            timeHistory['external_work'].append(external_work_global)
            timeHistory['strain_energy_adj'].append(strain_energy_adj_global.item()) 
            timeHistory['spring_energy_adj'].append(spring_energy_adj_global.item()) 
            timeHistory['external_work_adj'].append(external_work_adj_global.item()) 
            timeHistory['heat_energy_adj'].append(heat_energy_adj_global.item())
            timeHistory['source_energy_adj'].append(source_energy_adj_global.item())
            timeHistory['wc'].append(wc) 
            timeHistory['wd'].append(wd) 
            timeHistory['wm'].append(wm) 
            timeHistory['wp'].append(wp) 
            timeHistory['wtemp'].append(wtemp) 
            timeHistory['grey'].append(grey_fraction.detach().cpu().tolist())
            timeHistory['p'].append(model_list[0].MP['p']) 

            # --- checkpoint (rank 0 only) ---
            if  (epoch+1) % plotting_interval_TO == 0:
                end_time = time.time() 
                torch.save(
                    {
                        'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),
                        'model_u_adj_state_dict': model_list[1].mean_module_NN_All.state_dict(),
                        'model_T_state_dict': model_list[2].mean_module_NN_All.state_dict(),
                        'model_T_adj_state_dict': model_list[3].mean_module_NN_All.state_dict(),
                        'model_rho_state_dict': model_list[4].mean_module_NN_All.state_dict(),
                    }, 
                     save_folder + f'Trained_models_{epoch}.pth'
                    )
                total_time = total_time + (end_time - start_time)
                start_time = time.time()

            model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
            model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
            model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])

        end_time = time.time()
        total_time += (end_time - start_time)
        print(f"Run {i} finished in {total_time:.2f}s")

        # -----------------------------
        # Save metadata and results
        # -----------------------------
        torch.save(MP, save_folder + "MP.pt")
        with open(save_folder + "NN_config_disp.json", "w") as file:
            json.dump(NN_config_disp, file, indent=4)
        with open(save_folder + "NN_config_rho.json", "w") as file:
            json.dump(NN_config_rho, file, indent=4)

        torch.save(train_data, save_folder + "train_data.pth")

        with open(save_folder + "timeHistory.json", "w") as file:
            json.dump(timeHistory, file)

        result_file_path = f"{save_folder}Results_summary.txt"
        with open(result_file_path, 'w') as f:
            f.write(f"The total training time in second is: {total_time}\n")
            f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
            f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")

# ========================
# Launch block (torchrun)
# ========================
if __name__ == "__main__":
    # Try to get ranks from torchrun, else default to single-GPU debug mode
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ############################### Define Parameters ##############################################
    # base_folder = f"/home/alexsunuci/PIGP2D_Dec5th_2025/"
    # base_folder = f'C:/Users/Alex/Desktop/PIGP2D_Dec5th_2025/'
    base_folder = os.getcwd()
    random_state = [1,3,5,7,9,11,13,15,17,19]
    num_CP = 51
    Example = 'EX2D2'
    Case = 'AlCuFe_s_20K_1'
    Diff_type = 'CPS4_reduced'
    design_flag = False # wether we have design constraint or not
    N_worker = 1  # for local debugging with 1 GPU
    # define material properties and
    # wt = [1.0, 1.0, 1.0, 1.0] # full integration
    wt = [4.0] # reduced integration
    tau = 0.5 # softmax sharpness parameters
    T_inf = 293 # surrounding temperature K
    T0 = 473  # surface reference temperature K for convection BC
    TD = 673  # surface temperature BC K
    hs = 2.0e-8 # surface convection coefficient:  W/(um^2K)
    hv = 1.33e-9 # surface convection coefficient:  W/(um^2K)
    s = [0.0,-4.5e-8,-4.5e-8,-4.5e-8] # simulated body heat source for each material phase (positive means heat generation; otherwise means heat sink from the body)
    kappa = [1.0e-8, 23.7e-5, 40e-5, 6e-5] # thermal conductivity: W/(K um)   Ni, steel, Al, Copper: 9.07e-5, 6e-5, 23.7e-5, 40e-5, 2.59e-5
    alpha = [1.2e-5, 2.3e-5, 1.7e-5, 1.2e-5,] # thermal expansion coefficient: Ni, steel, Al, Copper, ti: 1.5e-5, 1.2e-5, 2.3e-5, 1.7e-5, 0.86e-5
    deltaT = 100 # isothermal temperature increment 
    D = np.array([0, 2.7, 8.96, 7.8])  # density g/cm^3: Ni, steel, Al, ti, Copper: 8.9, 7.8, 2.7, 8.96, 4.5
    D = D/D.max() # normalize D with the maximum value
    E = [1e-5,0.07,0.128,0.2]  # modulus GP/1000: Ni, steel, Al, Copper, ti: 0.2, 0.2, 0.07, 0.128, 0.12
    P = [0, 3.0, 2.0, 1.0]  # cost
    K_out = 2e-3 # magnitude of output spring
    f_out = 0.1 # magnitude of adjoint external force, the sign is handled in the data
    a_T = 1.0 
    nu = 0.31  # Poisson's ratio
    p0 = 3  # initial penalty
    pf = 3  # final penalty
    massfrac_f = 0.25  # final mass fraction
    costfrac_f = 0.6  # final mass fraction
    frac_decrease = 0.5  # fraction of epoch to decrease volume fraction from VF0 to VF_f
    b = 8  # sharpness parameters
    thres_static = 0.5  # static threshold values to binarize the density field
    rho_min = 0.1  # lower limit to define grey element (rho_min,rho_max)
    rho_max = 0.9  # upper limit to define grey element (rho_min,rho_max)

    # define domain
    thickness = 15.0
    Nelx = 500
    Nely = 250
    xmin, xmax = 0.0, 500.0
    ymin, ymax = 0.0, 250.0

    # define model parameters and training parameters
    init_method = 'kaiming_uniform_'
    dynamic_weight = False
    gradient_clip = False
    omega = 0.25
    learning_rate_disp = 1e-3
    learning_rate_disp_adj = 1e-3
    learning_rate_T = 1e-3
    learning_rate_T_adj = 1e-3
    learning_rate_rho = 1e-4
    nrmThreshold = 0.1
    TO_num_iter = 20000
    plot_num = 10
    delta = 1e-1
    wc = 1
    wd = 1
    wm = 1e2
    wp = 1e2
    wtemp = 1
    basis = 'PGCAN'
    basis_rho = 'PGCAN'
    quant_correlation_class = 'Rough_RBF'
    activation = 'tanh'
    if basis == 'PGCAN':
        n_features = 64
        n_cells = 3
        res = [24, 48]
        n_neurons = int(n_features / 2)
        n_layers = 3
        NN_arch = [n_neurons] * n_layers
        kernel_size = (2, 2)
    else:
        n_features = None
        n_cells = None
        res = None
        n_neurons = 64
        n_layers = 6
        NN_arch = [n_neurons] * n_layers
        kernel_size = None

    ############################### Pre-processing ##############################################
    num_phase = len(D)
    tkwargs = get_multiGPU(N_worker)
    D_tensor = [torch.tensor(D, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    E_tensor = [torch.tensor(E, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    P_tensor = [torch.tensor(P, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    alpha_tensor = [torch.tensor(alpha, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    kappa_tensor = [torch.tensor(kappa, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    s_tensor = [torch.tensor(s, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    wt_tensor = [torch.tensor(wt, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    domain = {'x': [xmin, xmax], 'y': [ymin, ymax], 'thickness':thickness}
    frac_decrease = int(frac_decrease * TO_num_iter)
    plotting_interval_TO = TO_num_iter // plot_num
    frac_step = None

    MP = {'num_CP': num_CP, 'Example': Example, 'N_worker': N_worker,'tau':tau,
          'base_folder': base_folder, 'domain': domain, 'design_flag':design_flag,
          'thres_static': thres_static, 'Diff_type': Diff_type,
          'num_phase': num_phase, 'Nelx': Nelx, 'Nely': Nely,
          'T_inf':T_inf,'T0':T0,'TD':TD,'hs':hs,'hv':hv,'s':s_tensor,
          'D': D_tensor, 'E': E_tensor, 'P': P_tensor, 'wt':wt_tensor,'nu': nu,'K_out':K_out,'f_out':f_out,
          'p': p0, 'p0': p0, 'pf': pf, 'b': b,'alpha':alpha_tensor,'kappa':kappa_tensor,'deltaT':deltaT,
          'rho_min': rho_min, 'rho_max': rho_max,'a_T':a_T,
          'massfrac_star': massfrac_f, 'massfrac0': massfrac_f,
          'massfrac_f': massfrac_f, 'frac_decrease': frac_decrease,
          'costfrac_star': costfrac_f, 'costfrac0': costfrac_f,
          'costfrac_f': costfrac_f,
          'frac_step_mass': frac_step, 'frac_step_cost': frac_step,
          'frac_step_p': frac_step}

    NN_config_disp = {'init_method': init_method, 'random_state': random_state, 'state': [],
                      'Example': Example, 'Case': Case,
                      'dynamic_weight': dynamic_weight, 'gradient_clip': gradient_clip,
                      'learning_rate_disp': learning_rate_disp,'learning_rate_disp_adj': learning_rate_disp_adj,
                      'learning_rate_rho': learning_rate_rho, 'learning_rate_T':learning_rate_T, 'learning_rate_T_adj':learning_rate_T_adj, 
                      'nrmThreshold': nrmThreshold, 'omega': omega,
                      'TO_num_iter': TO_num_iter, 'plot_num': plot_num,
                      'plotting_interval_TO': plotting_interval_TO,
                      'delta': delta, 'wc': wc, 'wd': wd, 'wm': wm, 'wp': wp,'wtemp':wtemp,
                      'kernel_size': kernel_size,
                      'basis': basis, 'quant_correlation_class': quant_correlation_class,
                      'activation': activation,
                      'n_features': n_features, 'n_cells': n_cells,
                      'res': res, 'NN_arch': NN_arch, 'save_folder': []}
    NN_config_rho = copy.deepcopy(NN_config_disp)
    NN_config_rho['basis'] = basis_rho
    if basis_rho == 'PGCAN':
        NN_config_rho['n_features'] = 64
        NN_config_rho['n_cells'] = 3
        NN_config_rho['res'] = [24, 48]
        n_neurons = int(NN_config_rho['n_features'] / 2)
        n_layers = 3
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers
        NN_config_rho['kernel_size'] = (2, 2)
    else:
        NN_config_rho['n_features'] = None
        NN_config_rho['n_cells'] = None
        NN_config_rho['res'] = None
        n_neurons = 64
        n_layers = 6
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers

    ############################### Generate Data ##############################################
    Training, mesh_data = get_data(MP)
    u_X_train = Training['u_X_train']
    v_X_train = Training['v_X_train']
    T_X_train = Training['T_X_train']
    u_train = Training['u_train']
    v_train = Training['v_train']
    T_train = Training['T_train']
    vd1_X_train = Training['vd1_X_train']
    vd1_train = Training['vd1_train']
    vd2_X_train = Training['vd2_X_train']
    vd2_train = Training['vd2_train']
    vt_X_train = Training['vt_X_train']
    vt_train = Training['vt_train']

    # create dict for trainign data
    train_data = {
                'disp': {
                    'X': [u_X_train, v_X_train],
                    'y': [u_train, v_train],
                },
                'disp_adj': {
                    'X': [vd1_X_train, vd2_X_train],
                    'y': [vd1_train, vd2_train],
                },
                'T': {
                    'X': [T_X_train],
                    'y': [T_train],
                },
                'T_adj': {
                    'X': [vt_X_train],
                    'y': [vt_train],
                },
                'phase': {
                    'X': [Training[f'phase{i}_X_train'] for i in range(num_phase)],
                    'y': [Training[f'phase{i}_train']   for i in range(num_phase)],
                },
            }
    
    # Generate random integers
    index_CP = np.random.randint(1, num_CP + 1, size=TO_num_iter)

    # Call worker directly (torchrun handles ranks)
    run_worker(MP, NN_config_disp, NN_config_rho,train_data,
               index_CP, mesh_data)

