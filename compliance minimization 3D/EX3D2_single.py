import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import time
import shutil
from models.lmgp import LMGP 
from tqdm import tqdm
from utils.utils_general import set_seed0, get_tkwargs
from gpytorch.settings import cholesky_jitter
from utils.get_training_data_3D import get_data_EX3D2 as get_data

tkwargs = get_tkwargs()

def eval_initial_massfrac(model_list):
    D = model_list[0].MP['D']  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    dz = model_list[0].dz
    elem_volume = dx * dy * dz
    
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()
    
    for model in model_list:
        model.eval()
    
    # Output raw phase logits
    m_phase = model_list[3].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
    # Apply softmax across phase dimension (dim=1)
    phase_weights = F.softmax(m_phase, dim=1)  # shape: [n_elem, n_phases]
    
    # Compute interpolated density
    phase_density = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    
    # Apply mask to select only relevant elements
    phase_density_masked = phase_density[mask_col_elem]  # shape: [n_masked_elem]
    
    # Total mass = sum of density × element volume
    mass = (phase_density_masked * elem_volume).sum()
    M0 = (torch.ones_like(phase_density_masked) * elem_volume).sum()
    massfrac = mass / M0
    return massfrac.item()

def eval_initial_costfrac(model_list):
    P = model_list[0].MP['P']  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    dz = model_list[0].dz
    elem_volume = dx * dy * dz
    
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()
    
    for model in model_list:
        model.eval()
    
    # Output raw phase logits
    m_phase = model_list[3].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
    # Apply softmax across phase dimension (dim=1)
    phase_weights = F.softmax(m_phase, dim=1)  # shape: [n_elem, n_phases]
    
    # Compute interpolated density
    phase_cost = (phase_weights * P).sum(dim=1)  # shape: [n_elem]
    
    # Apply mask to select only relevant elements
    phase_cost_masked = phase_cost[mask_col_elem]  # shape: [n_masked_elem]
    
    # Total mass = sum of density × element volume
    cost = (phase_cost_masked * elem_volume).sum()
    cost0 = (P.max()*torch.ones_like(phase_cost_masked) * elem_volume).sum()
    costfrac = cost / cost0
    return costfrac.item()

def calculate_TO_loss(model_list,Diff_type):
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D']  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    P = model_list[0].MP['P']  # e.g., [0, 1.6, 1.2, 1.0] a tensor
    E_tensor = model_list[0].MP['E'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    dz = model_list[0].dz
    elem_volume = dx * dy * dz

    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    massfrac_star = model_list[0].MP['massfrac_star']
    costfrac_star = model_list[0].MP['costfrac_star']
    Nx = model_list[0].Nx
    Ny = model_list[0].Ny
    Nz = model_list[0].Nz

    mask_col_elem = model_list[0].mask_col_elem
    collocation_x = model_list[0].collocation_x.clone()
    elem_x = model_list[0].elem_x.clone()

    for model in model_list:
        model.train

    m_col = model_list[0].mean_module_NN_All(collocation_x)
    m_phase = model_list[3].mean_module_NN_All(elem_x)

    g_u = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()
    g_v = (model_list[1].covar_module(model_list[1].train_inputs[0], collocation_x)).evaluate()
    g_w = (model_list[2].covar_module(model_list[2].train_inputs[0], collocation_x)).evaluate()
    # g_rho = (model_list[3].covar_module(model_list[3].train_inputs[0], elem_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()
            model_list[1].chol_decomp = model_list[1].covar_module(model_list[1].train_inputs[0]).cholesky()
            model_list[2].chol_decomp = model_list[2].covar_module(model_list[2].train_inputs[0]).cholesky()
            # model_list[3].chol_decomp = model_list[3].covar_module(model_list[3].train_inputs[0]).cholesky()

    K_inv_offset_u = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))
    K_inv_offset_v = model_list[1].chol_decomp._cholesky_solve(model_list[1].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[1].train_inputs[0])[:,1].unsqueeze(-1))
    K_inv_offset_w = model_list[2].chol_decomp._cholesky_solve(model_list[2].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[2].train_inputs[0])[:,2].unsqueeze(-1))
    # K_inv_offset_rho = model_list[3].chol_decomp._cholesky_solve(model_list[3].train_targets.unsqueeze(-1) - torch.sigmoid(model_list[3].mean_module_NN_All(model_list[3].train_inputs[0])[:,0].unsqueeze(-1)))

    u = (m_col[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u).squeeze(-1)
    v = (m_col[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v).squeeze(-1)
    w = (m_col[:,2].unsqueeze(-1) + g_w.t() @ K_inv_offset_w).squeeze(-1)
    
    # Apply softmax across phase dimension (dim=1)
    phase_weights = F.softmax(m_phase, dim=1)  # shape: [n_elem, n_phases]
    phase_weights_masked = phase_weights[mask_col_elem] # shape: [n_masked_elem, n_phases]
    phase_density_masked = (phase_weights_masked * D).sum(dim=1)  # shape: [n_masked_elem]
    mass = (phase_density_masked * elem_volume).sum()
    M0 = (torch.ones_like(phase_density_masked) * elem_volume).sum()
    massfrac = mass / M0
    loss_mConstraint = torch.square(torch.max((massfrac / massfrac_star) - 1.0, 0.0*mass))
    # loss_mConstraint = torch.square(torch.clamp((massfrac / massfrac_star) - 1.0, min=0.0))

    # Compute interpolated density
    phase_cost_masked = (phase_weights_masked * P).sum(dim=1)  # shape: [n_elem]
    cost = (phase_cost_masked * elem_volume).sum()
    cost0 = (P.max()*torch.ones_like(phase_cost_masked) * elem_volume).sum()
    costfrac = cost / cost0
    loss_cConstraint = torch.square(torch.max((costfrac / costfrac_star) - 1.0, 0.0*cost))
    # loss_cConstraint = torch.square(torch.clamp((costfrac / costfrac_star) - 1.0, min=0.0))

    # Calculate fraction of gray elements per phase using NumPy
    grey_fraction = []
    weights_np = phase_weights_masked.detach().cpu().numpy()  # convert to NumPy
    for i in range(num_phase):
        grey_elements = (weights_np[:, i] > MP['rho_min']) & (weights_np[:, i] < MP['rho_max'])
        grey_fraction.append(np.sum(grey_elements) / len(weights_np))
        
    # calculate the external work
    v_load = v[model_list[0].traction_indices].squeeze(-1)
    external_work = torch.sum(v_load * model_list[0].traction_magnitude)
    
    # calculate the young's modulus
    phase_weights_no_grad = phase_weights.detach() # shape: [n_elem, n_phases]
    E_node = (E_tensor * phase_weights_no_grad ** p).sum(dim = 1) # shape: [n_elem]
    E_hat_node = (E_tensor * (phase_weights_no_grad)**(2*p)/((phase_weights + 1e-8) ** p)).sum(dim = 1) # shape: [n_elem]
    E = E_node.reshape((Nx-1),(Ny-1),(Nz-1)) 
    E_hat = E_hat_node.reshape((Nx-1),(Ny-1),(Nz-1)) 

    lambdda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    lambdda_hat = E_hat*nu/(1+nu)/(1-2*nu)
    mu_hat = E_hat/2/(1+nu)

    if Diff_type == 'AD': #autograd
        pass
    elif Diff_type == 'hex': 
        detJ = model_list[0].detJ
        dN = model_list[0].dN
        dN1_GP1 = dN[0]
        dN1_GP2 = dN[1]
        dN1_GP3 = dN[2]
        dN1_GP4 = dN[3]
        dN1_GP5 = dN[4]
        dN1_GP6 = dN[5]
        dN1_GP7 = dN[6]
        dN1_GP8 = dN[7]

        dN2_GP1 = dN[8]
        dN2_GP2 = dN[9]
        dN2_GP3 = dN[10]
        dN2_GP4 = dN[11]
        dN2_GP5 = dN[12]
        dN2_GP6 = dN[13]
        dN2_GP7 = dN[14]
        dN2_GP8 = dN[15]

        dN3_GP1 = dN[16]
        dN3_GP2 = dN[17]
        dN3_GP3 = dN[18]
        dN3_GP4 = dN[19]
        dN3_GP5 = dN[20]
        dN3_GP6 = dN[21]
        dN3_GP7 = dN[22]
        dN3_GP8 = dN[23]

        dN4_GP1 = dN[24]
        dN4_GP2 = dN[25]
        dN4_GP3 = dN[26]
        dN4_GP4 = dN[27]
        dN4_GP5 = dN[28]
        dN4_GP6 = dN[29]
        dN4_GP7 = dN[30]
        dN4_GP8 = dN[31]

        dN5_GP1 = dN[32]
        dN5_GP2 = dN[33]
        dN5_GP3 = dN[34]
        dN5_GP4 = dN[35]
        dN5_GP5 = dN[36]
        dN5_GP6 = dN[37]
        dN5_GP7 = dN[38]
        dN5_GP8 = dN[39]

        dN6_GP1 = dN[40]
        dN6_GP2 = dN[41]
        dN6_GP3 = dN[42]
        dN6_GP4 = dN[43]
        dN6_GP5 = dN[44]
        dN6_GP6 = dN[45]
        dN6_GP7 = dN[46]
        dN6_GP8 = dN[47]

        dN7_GP1 = dN[48]
        dN7_GP2 = dN[49]
        dN7_GP3 = dN[50]
        dN7_GP4 = dN[51]
        dN7_GP5 = dN[52]
        dN7_GP6 = dN[53]
        dN7_GP7 = dN[54]
        dN7_GP8 = dN[55]

        dN8_GP1 = dN[56]
        dN8_GP2 = dN[57]
        dN8_GP3 = dN[58]
        dN8_GP4 = dN[59]
        dN8_GP5 = dN[60]
        dN8_GP6 = dN[61]
        dN8_GP7 = dN[62]
        dN8_GP8 = dN[63]

        # dN = [dN1_GP1, dN1_GP2, dN1_GP3, dN1_GP4, dN1_GP5, dN1_GP6, dN1_GP7, dN1_GP8,
        #       dN2_GP1, dN2_GP2, dN2_GP3, dN2_GP4, dN2_GP5, dN2_GP6, dN2_GP7, dN2_GP8,
        #       dN3_GP1, dN3_GP2, dN3_GP3, dN3_GP4, dN3_GP5, dN3_GP6, dN3_GP7, dN3_GP8,
        #       dN4_GP1, dN4_GP2, dN4_GP3, dN4_GP4, dN4_GP5, dN4_GP6, dN4_GP7, dN4_GP8,
        #       dN5_GP1, dN5_GP2, dN5_GP3, dN5_GP4, dN5_GP5, dN5_GP6, dN5_GP7, dN5_GP8,
        #       dN6_GP1, dN6_GP2, dN6_GP3, dN6_GP4, dN6_GP5, dN6_GP6, dN6_GP7, dN6_GP8,
        #       dN7_GP1, dN7_GP2, dN7_GP3, dN7_GP4, dN7_GP5, dN7_GP6, dN7_GP7, dN7_GP8,
        #       dN8_GP1, dN8_GP2, dN8_GP3, dN8_GP4, dN8_GP5, dN8_GP6, dN8_GP7, dN8_GP8,]

        Ux= u.reshape(Nx, Ny, Nz)
        Uy= v.reshape(Nx, Ny, Nz)
        Uz= w.reshape(Nx, Ny, Nz)

        UxN1= Ux[:-1,:-1,:-1]
        UxN2= Ux[1:,:-1,:-1]
        UxN3= Ux[1:,1:,:-1]
        UxN4= Ux[:-1,1:,:-1]
        UxN5= Ux[:-1,:-1,1:]
        UxN6= Ux[1:,:-1,1:]
        UxN7= Ux[1:,1:,1:]
        UxN8= Ux[:-1,1:,1:]

        UyN1= Uy[:-1,:-1,:-1]
        UyN2= Uy[1:,:-1,:-1]
        UyN3= Uy[1:,1:,:-1]
        UyN4= Uy[:-1,1:,:-1]
        UyN5= Uy[:-1,:-1,1:]
        UyN6= Uy[1:,:-1,1:]
        UyN7= Uy[1:,1:,1:]
        UyN8= Uy[:-1,1:,1:]

        UzN1= Uz[:-1,:-1,:-1]
        UzN2= Uz[1:,:-1,:-1]
        UzN3= Uz[1:,1:,:-1]
        UzN4= Uz[:-1,1:,:-1]
        UzN5= Uz[:-1,:-1,1:]
        UzN6= Uz[1:,:-1,1:]
        UzN7= Uz[1:,1:,1:]
        UzN8= Uz[:-1,1:,1:]

        dUxdx_GP1= dN1_GP1[0][0]*UxN1 + dN2_GP1[0][0]*UxN2 + dN3_GP1[0][0]*UxN3 + dN4_GP1[0][0]*UxN4 + dN5_GP1[0][0]*UxN5 + dN6_GP1[0][0]*UxN6 + dN7_GP1[0][0]*UxN7 + dN8_GP1[0][0]*UxN8
        dUxdy_GP1= dN1_GP1[1][0]*UxN1 + dN2_GP1[1][0]*UxN2 + dN3_GP1[1][0]*UxN3 + dN4_GP1[1][0]*UxN4 + dN5_GP1[1][0]*UxN5 + dN6_GP1[1][0]*UxN6 + dN7_GP1[1][0]*UxN7 + dN8_GP1[1][0]*UxN8
        dUxdz_GP1= dN1_GP1[2][0]*UxN1 + dN2_GP1[2][0]*UxN2 + dN3_GP1[2][0]*UxN3 + dN4_GP1[2][0]*UxN4 + dN5_GP1[2][0]*UxN5 + dN6_GP1[2][0]*UxN6 + dN7_GP1[2][0]*UxN7 + dN8_GP1[2][0]*UxN8
        dUydx_GP1= dN1_GP1[0][0]*UyN1 + dN2_GP1[0][0]*UyN2 + dN3_GP1[0][0]*UyN3 + dN4_GP1[0][0]*UyN4 + dN5_GP1[0][0]*UyN5 + dN6_GP1[0][0]*UyN6 + dN7_GP1[0][0]*UyN7 + dN8_GP1[0][0]*UyN8
        dUydy_GP1= dN1_GP1[1][0]*UyN1 + dN2_GP1[1][0]*UyN2 + dN3_GP1[1][0]*UyN3 + dN4_GP1[1][0]*UyN4 + dN5_GP1[1][0]*UyN5 + dN6_GP1[1][0]*UyN6 + dN7_GP1[1][0]*UyN7 + dN8_GP1[1][0]*UyN8
        dUydz_GP1= dN1_GP1[2][0]*UyN1 + dN2_GP1[2][0]*UyN2 + dN3_GP1[2][0]*UyN3 + dN4_GP1[2][0]*UyN4 + dN5_GP1[2][0]*UyN5 + dN6_GP1[2][0]*UyN6 + dN7_GP1[2][0]*UyN7 + dN8_GP1[2][0]*UyN8
        dUzdx_GP1= dN1_GP1[0][0]*UzN1 + dN2_GP1[0][0]*UzN2 + dN3_GP1[0][0]*UzN3 + dN4_GP1[0][0]*UzN4 + dN5_GP1[0][0]*UzN5 + dN6_GP1[0][0]*UzN6 + dN7_GP1[0][0]*UzN7 + dN8_GP1[0][0]*UzN8
        dUzdy_GP1= dN1_GP1[1][0]*UzN1 + dN2_GP1[1][0]*UzN2 + dN3_GP1[1][0]*UzN3 + dN4_GP1[1][0]*UzN4 + dN5_GP1[1][0]*UzN5 + dN6_GP1[1][0]*UzN6 + dN7_GP1[1][0]*UzN7 + dN8_GP1[1][0]*UzN8
        dUzdz_GP1= dN1_GP1[2][0]*UzN1 + dN2_GP1[2][0]*UzN2 + dN3_GP1[2][0]*UzN3 + dN4_GP1[2][0]*UzN4 + dN5_GP1[2][0]*UzN5 + dN6_GP1[2][0]*UzN6 + dN7_GP1[2][0]*UzN7 + dN8_GP1[2][0]*UzN8

        dUxdx_GP2= dN1_GP2[0][0]*UxN1 + dN2_GP2[0][0]*UxN2 + dN3_GP2[0][0]*UxN3 + dN4_GP2[0][0]*UxN4 + dN5_GP2[0][0]*UxN5 + dN6_GP2[0][0]*UxN6 + dN7_GP2[0][0]*UxN7 + dN8_GP2[0][0]*UxN8
        dUxdy_GP2= dN1_GP2[1][0]*UxN1 + dN2_GP2[1][0]*UxN2 + dN3_GP2[1][0]*UxN3 + dN4_GP2[1][0]*UxN4 + dN5_GP2[1][0]*UxN5 + dN6_GP2[1][0]*UxN6 + dN7_GP2[1][0]*UxN7 + dN8_GP2[1][0]*UxN8
        dUxdz_GP2= dN1_GP2[2][0]*UxN1 + dN2_GP2[2][0]*UxN2 + dN3_GP2[2][0]*UxN3 + dN4_GP2[2][0]*UxN4 + dN5_GP2[2][0]*UxN5 + dN6_GP2[2][0]*UxN6 + dN7_GP2[2][0]*UxN7 + dN8_GP2[2][0]*UxN8
        dUydx_GP2= dN1_GP2[0][0]*UyN1 + dN2_GP2[0][0]*UyN2 + dN3_GP2[0][0]*UyN3 + dN4_GP2[0][0]*UyN4 + dN5_GP2[0][0]*UyN5 + dN6_GP2[0][0]*UyN6 + dN7_GP2[0][0]*UyN7 + dN8_GP2[0][0]*UyN8
        dUydy_GP2= dN1_GP2[1][0]*UyN1 + dN2_GP2[1][0]*UyN2 + dN3_GP2[1][0]*UyN3 + dN4_GP2[1][0]*UyN4 + dN5_GP2[1][0]*UyN5 + dN6_GP2[1][0]*UyN6 + dN7_GP2[1][0]*UyN7 + dN8_GP2[1][0]*UyN8
        dUydz_GP2= dN1_GP2[2][0]*UyN1 + dN2_GP2[2][0]*UyN2 + dN3_GP2[2][0]*UyN3 + dN4_GP2[2][0]*UyN4 + dN5_GP2[2][0]*UyN5 + dN6_GP2[2][0]*UyN6 + dN7_GP2[2][0]*UyN7 + dN8_GP2[2][0]*UyN8
        dUzdx_GP2= dN1_GP2[0][0]*UzN1 + dN2_GP2[0][0]*UzN2 + dN3_GP2[0][0]*UzN3 + dN4_GP2[0][0]*UzN4 + dN5_GP2[0][0]*UzN5 + dN6_GP2[0][0]*UzN6 + dN7_GP2[0][0]*UzN7 + dN8_GP2[0][0]*UzN8
        dUzdy_GP2= dN1_GP2[1][0]*UzN1 + dN2_GP2[1][0]*UzN2 + dN3_GP2[1][0]*UzN3 + dN4_GP2[1][0]*UzN4 + dN5_GP2[1][0]*UzN5 + dN6_GP2[1][0]*UzN6 + dN7_GP2[1][0]*UzN7 + dN8_GP2[1][0]*UzN8
        dUzdz_GP2= dN1_GP2[2][0]*UzN1 + dN2_GP2[2][0]*UzN2 + dN3_GP2[2][0]*UzN3 + dN4_GP2[2][0]*UzN4 + dN5_GP2[2][0]*UzN5 + dN6_GP2[2][0]*UzN6 + dN7_GP2[2][0]*UzN7 + dN8_GP2[2][0]*UzN8

        dUxdx_GP3= dN1_GP3[0][0]*UxN1 + dN2_GP3[0][0]*UxN2 + dN3_GP3[0][0]*UxN3 + dN4_GP3[0][0]*UxN4 + dN5_GP3[0][0]*UxN5 + dN6_GP3[0][0]*UxN6 + dN7_GP3[0][0]*UxN7 + dN8_GP3[0][0]*UxN8
        dUxdy_GP3= dN1_GP3[1][0]*UxN1 + dN2_GP3[1][0]*UxN2 + dN3_GP3[1][0]*UxN3 + dN4_GP3[1][0]*UxN4 + dN5_GP3[1][0]*UxN5 + dN6_GP3[1][0]*UxN6 + dN7_GP3[1][0]*UxN7 + dN8_GP3[1][0]*UxN8
        dUxdz_GP3= dN1_GP3[2][0]*UxN1 + dN2_GP3[2][0]*UxN2 + dN3_GP3[2][0]*UxN3 + dN4_GP3[2][0]*UxN4 + dN5_GP3[2][0]*UxN5 + dN6_GP3[2][0]*UxN6 + dN7_GP3[2][0]*UxN7 + dN8_GP3[2][0]*UxN8
        dUydx_GP3= dN1_GP3[0][0]*UyN1 + dN2_GP3[0][0]*UyN2 + dN3_GP3[0][0]*UyN3 + dN4_GP3[0][0]*UyN4 + dN5_GP3[0][0]*UyN5 + dN6_GP3[0][0]*UyN6 + dN7_GP3[0][0]*UyN7 + dN8_GP3[0][0]*UyN8
        dUydy_GP3= dN1_GP3[1][0]*UyN1 + dN2_GP3[1][0]*UyN2 + dN3_GP3[1][0]*UyN3 + dN4_GP3[1][0]*UyN4 + dN5_GP3[1][0]*UyN5 + dN6_GP3[1][0]*UyN6 + dN7_GP3[1][0]*UyN7 + dN8_GP3[1][0]*UyN8
        dUydz_GP3= dN1_GP3[2][0]*UyN1 + dN2_GP3[2][0]*UyN2 + dN3_GP3[2][0]*UyN3 + dN4_GP3[2][0]*UyN4 + dN5_GP3[2][0]*UyN5 + dN6_GP3[2][0]*UyN6 + dN7_GP3[2][0]*UyN7 + dN8_GP3[2][0]*UyN8
        dUzdx_GP3= dN1_GP3[0][0]*UzN1 + dN2_GP3[0][0]*UzN2 + dN3_GP3[0][0]*UzN3 + dN4_GP3[0][0]*UzN4 + dN5_GP3[0][0]*UzN5 + dN6_GP3[0][0]*UzN6 + dN7_GP3[0][0]*UzN7 + dN8_GP3[0][0]*UzN8
        dUzdy_GP3= dN1_GP3[1][0]*UzN1 + dN2_GP3[1][0]*UzN2 + dN3_GP3[1][0]*UzN3 + dN4_GP3[1][0]*UzN4 + dN5_GP3[1][0]*UzN5 + dN6_GP3[1][0]*UzN6 + dN7_GP3[1][0]*UzN7 + dN8_GP3[1][0]*UzN8
        dUzdz_GP3= dN1_GP3[2][0]*UzN1 + dN2_GP3[2][0]*UzN2 + dN3_GP3[2][0]*UzN3 + dN4_GP3[2][0]*UzN4 + dN5_GP3[2][0]*UzN5 + dN6_GP3[2][0]*UzN6 + dN7_GP3[2][0]*UzN7 + dN8_GP3[2][0]*UzN8

        dUxdx_GP4= dN1_GP4[0][0]*UxN1 + dN2_GP4[0][0]*UxN2 + dN3_GP4[0][0]*UxN3 + dN4_GP4[0][0]*UxN4 + dN5_GP4[0][0]*UxN5 + dN6_GP4[0][0]*UxN6 + dN7_GP4[0][0]*UxN7 + dN8_GP4[0][0]*UxN8
        dUxdy_GP4= dN1_GP4[1][0]*UxN1 + dN2_GP4[1][0]*UxN2 + dN3_GP4[1][0]*UxN3 + dN4_GP4[1][0]*UxN4 + dN5_GP4[1][0]*UxN5 + dN6_GP4[1][0]*UxN6 + dN7_GP4[1][0]*UxN7 + dN8_GP4[1][0]*UxN8
        dUxdz_GP4= dN1_GP4[2][0]*UxN1 + dN2_GP4[2][0]*UxN2 + dN3_GP4[2][0]*UxN3 + dN4_GP4[2][0]*UxN4 + dN5_GP4[2][0]*UxN5 + dN6_GP4[2][0]*UxN6 + dN7_GP4[2][0]*UxN7 + dN8_GP4[2][0]*UxN8
        dUydx_GP4= dN1_GP4[0][0]*UyN1 + dN2_GP4[0][0]*UyN2 + dN3_GP4[0][0]*UyN3 + dN4_GP4[0][0]*UyN4 + dN5_GP4[0][0]*UyN5 + dN6_GP4[0][0]*UyN6 + dN7_GP4[0][0]*UyN7 + dN8_GP4[0][0]*UyN8
        dUydy_GP4= dN1_GP4[1][0]*UyN1 + dN2_GP4[1][0]*UyN2 + dN3_GP4[1][0]*UyN3 + dN4_GP4[1][0]*UyN4 + dN5_GP4[1][0]*UyN5 + dN6_GP4[1][0]*UyN6 + dN7_GP4[1][0]*UyN7 + dN8_GP4[1][0]*UyN8
        dUydz_GP4= dN1_GP4[2][0]*UyN1 + dN2_GP4[2][0]*UyN2 + dN3_GP4[2][0]*UyN3 + dN4_GP4[2][0]*UyN4 + dN5_GP4[2][0]*UyN5 + dN6_GP4[2][0]*UyN6 + dN7_GP4[2][0]*UyN7 + dN8_GP4[2][0]*UyN8
        dUzdx_GP4= dN1_GP4[0][0]*UzN1 + dN2_GP4[0][0]*UzN2 + dN3_GP4[0][0]*UzN3 + dN4_GP4[0][0]*UzN4 + dN5_GP4[0][0]*UzN5 + dN6_GP4[0][0]*UzN6 + dN7_GP4[0][0]*UzN7 + dN8_GP4[0][0]*UzN8
        dUzdy_GP4= dN1_GP4[1][0]*UzN1 + dN2_GP4[1][0]*UzN2 + dN3_GP4[1][0]*UzN3 + dN4_GP4[1][0]*UzN4 + dN5_GP4[1][0]*UzN5 + dN6_GP4[1][0]*UzN6 + dN7_GP4[1][0]*UzN7 + dN8_GP4[1][0]*UzN8
        dUzdz_GP4= dN1_GP4[2][0]*UzN1 + dN2_GP4[2][0]*UzN2 + dN3_GP4[2][0]*UzN3 + dN4_GP4[2][0]*UzN4 + dN5_GP4[2][0]*UzN5 + dN6_GP4[2][0]*UzN6 + dN7_GP4[2][0]*UzN7 + dN8_GP4[2][0]*UzN8

        dUxdx_GP5= dN1_GP5[0][0]*UxN1 + dN2_GP5[0][0]*UxN2 + dN3_GP5[0][0]*UxN3 + dN4_GP5[0][0]*UxN4 + dN5_GP5[0][0]*UxN5 + dN6_GP5[0][0]*UxN6 + dN7_GP5[0][0]*UxN7 + dN8_GP5[0][0]*UxN8
        dUxdy_GP5= dN1_GP5[1][0]*UxN1 + dN2_GP5[1][0]*UxN2 + dN3_GP5[1][0]*UxN3 + dN4_GP5[1][0]*UxN4 + dN5_GP5[1][0]*UxN5 + dN6_GP5[1][0]*UxN6 + dN7_GP5[1][0]*UxN7 + dN8_GP5[1][0]*UxN8
        dUxdz_GP5= dN1_GP5[2][0]*UxN1 + dN2_GP5[2][0]*UxN2 + dN3_GP5[2][0]*UxN3 + dN4_GP5[2][0]*UxN4 + dN5_GP5[2][0]*UxN5 + dN6_GP5[2][0]*UxN6 + dN7_GP5[2][0]*UxN7 + dN8_GP5[2][0]*UxN8
        dUydx_GP5= dN1_GP5[0][0]*UyN1 + dN2_GP5[0][0]*UyN2 + dN3_GP5[0][0]*UyN3 + dN4_GP5[0][0]*UyN4 + dN5_GP5[0][0]*UyN5 + dN6_GP5[0][0]*UyN6 + dN7_GP5[0][0]*UyN7 + dN8_GP5[0][0]*UyN8
        dUydy_GP5= dN1_GP5[1][0]*UyN1 + dN2_GP5[1][0]*UyN2 + dN3_GP5[1][0]*UyN3 + dN4_GP5[1][0]*UyN4 + dN5_GP5[1][0]*UyN5 + dN6_GP5[1][0]*UyN6 + dN7_GP5[1][0]*UyN7 + dN8_GP5[1][0]*UyN8
        dUydz_GP5= dN1_GP5[2][0]*UyN1 + dN2_GP5[2][0]*UyN2 + dN3_GP5[2][0]*UyN3 + dN4_GP5[2][0]*UyN4 + dN5_GP5[2][0]*UyN5 + dN6_GP5[2][0]*UyN6 + dN7_GP5[2][0]*UyN7 + dN8_GP5[2][0]*UyN8
        dUzdx_GP5= dN1_GP5[0][0]*UzN1 + dN2_GP5[0][0]*UzN2 + dN3_GP5[0][0]*UzN3 + dN4_GP5[0][0]*UzN4 + dN5_GP5[0][0]*UzN5 + dN6_GP5[0][0]*UzN6 + dN7_GP5[0][0]*UzN7 + dN8_GP5[0][0]*UzN8
        dUzdy_GP5= dN1_GP5[1][0]*UzN1 + dN2_GP5[1][0]*UzN2 + dN3_GP5[1][0]*UzN3 + dN4_GP5[1][0]*UzN4 + dN5_GP5[1][0]*UzN5 + dN6_GP5[1][0]*UzN6 + dN7_GP5[1][0]*UzN7 + dN8_GP5[1][0]*UzN8
        dUzdz_GP5= dN1_GP5[2][0]*UzN1 + dN2_GP5[2][0]*UzN2 + dN3_GP5[2][0]*UzN3 + dN4_GP5[2][0]*UzN4 + dN5_GP5[2][0]*UzN5 + dN6_GP5[2][0]*UzN6 + dN7_GP5[2][0]*UzN7 + dN8_GP5[2][0]*UzN8

        dUxdx_GP6= dN1_GP6[0][0]*UxN1 + dN2_GP6[0][0]*UxN2 + dN3_GP6[0][0]*UxN3 + dN4_GP6[0][0]*UxN4 + dN5_GP6[0][0]*UxN5 + dN6_GP6[0][0]*UxN6 + dN7_GP6[0][0]*UxN7 + dN8_GP6[0][0]*UxN8
        dUxdy_GP6= dN1_GP6[1][0]*UxN1 + dN2_GP6[1][0]*UxN2 + dN3_GP6[1][0]*UxN3 + dN4_GP6[1][0]*UxN4 + dN5_GP6[1][0]*UxN5 + dN6_GP6[1][0]*UxN6 + dN7_GP6[1][0]*UxN7 + dN8_GP6[1][0]*UxN8
        dUxdz_GP6= dN1_GP6[2][0]*UxN1 + dN2_GP6[2][0]*UxN2 + dN3_GP6[2][0]*UxN3 + dN4_GP6[2][0]*UxN4 + dN5_GP6[2][0]*UxN5 + dN6_GP6[2][0]*UxN6 + dN7_GP6[2][0]*UxN7 + dN8_GP6[2][0]*UxN8
        dUydx_GP6= dN1_GP6[0][0]*UyN1 + dN2_GP6[0][0]*UyN2 + dN3_GP6[0][0]*UyN3 + dN4_GP6[0][0]*UyN4 + dN5_GP6[0][0]*UyN5 + dN6_GP6[0][0]*UyN6 + dN7_GP6[0][0]*UyN7 + dN8_GP6[0][0]*UyN8
        dUydy_GP6= dN1_GP6[1][0]*UyN1 + dN2_GP6[1][0]*UyN2 + dN3_GP6[1][0]*UyN3 + dN4_GP6[1][0]*UyN4 + dN5_GP6[1][0]*UyN5 + dN6_GP6[1][0]*UyN6 + dN7_GP6[1][0]*UyN7 + dN8_GP6[1][0]*UyN8
        dUydz_GP6= dN1_GP6[2][0]*UyN1 + dN2_GP6[2][0]*UyN2 + dN3_GP6[2][0]*UyN3 + dN4_GP6[2][0]*UyN4 + dN5_GP6[2][0]*UyN5 + dN6_GP6[2][0]*UyN6 + dN7_GP6[2][0]*UyN7 + dN8_GP6[2][0]*UyN8
        dUzdx_GP6= dN1_GP6[0][0]*UzN1 + dN2_GP6[0][0]*UzN2 + dN3_GP6[0][0]*UzN3 + dN4_GP6[0][0]*UzN4 + dN5_GP6[0][0]*UzN5 + dN6_GP6[0][0]*UzN6 + dN7_GP6[0][0]*UzN7 + dN8_GP6[0][0]*UzN8
        dUzdy_GP6= dN1_GP6[1][0]*UzN1 + dN2_GP6[1][0]*UzN2 + dN3_GP6[1][0]*UzN3 + dN4_GP6[1][0]*UzN4 + dN5_GP6[1][0]*UzN5 + dN6_GP6[1][0]*UzN6 + dN7_GP6[1][0]*UzN7 + dN8_GP6[1][0]*UzN8
        dUzdz_GP6= dN1_GP6[2][0]*UzN1 + dN2_GP6[2][0]*UzN2 + dN3_GP6[2][0]*UzN3 + dN4_GP6[2][0]*UzN4 + dN5_GP6[2][0]*UzN5 + dN6_GP6[2][0]*UzN6 + dN7_GP6[2][0]*UzN7 + dN8_GP6[2][0]*UzN8

        dUxdx_GP7= dN1_GP7[0][0]*UxN1 + dN2_GP7[0][0]*UxN2 + dN3_GP7[0][0]*UxN3 + dN4_GP7[0][0]*UxN4 + dN5_GP7[0][0]*UxN5 + dN6_GP7[0][0]*UxN6 + dN7_GP7[0][0]*UxN7 + dN8_GP7[0][0]*UxN8
        dUxdy_GP7= dN1_GP7[1][0]*UxN1 + dN2_GP7[1][0]*UxN2 + dN3_GP7[1][0]*UxN3 + dN4_GP7[1][0]*UxN4 + dN5_GP7[1][0]*UxN5 + dN6_GP7[1][0]*UxN6 + dN7_GP7[1][0]*UxN7 + dN8_GP7[1][0]*UxN8
        dUxdz_GP7= dN1_GP7[2][0]*UxN1 + dN2_GP7[2][0]*UxN2 + dN3_GP7[2][0]*UxN3 + dN4_GP7[2][0]*UxN4 + dN5_GP7[2][0]*UxN5 + dN6_GP7[2][0]*UxN6 + dN7_GP7[2][0]*UxN7 + dN8_GP7[2][0]*UxN8
        dUydx_GP7= dN1_GP7[0][0]*UyN1 + dN2_GP7[0][0]*UyN2 + dN3_GP7[0][0]*UyN3 + dN4_GP7[0][0]*UyN4 + dN5_GP7[0][0]*UyN5 + dN6_GP7[0][0]*UyN6 + dN7_GP7[0][0]*UyN7 + dN8_GP7[0][0]*UyN8
        dUydy_GP7= dN1_GP7[1][0]*UyN1 + dN2_GP7[1][0]*UyN2 + dN3_GP7[1][0]*UyN3 + dN4_GP7[1][0]*UyN4 + dN5_GP7[1][0]*UyN5 + dN6_GP7[1][0]*UyN6 + dN7_GP7[1][0]*UyN7 + dN8_GP7[1][0]*UyN8
        dUydz_GP7= dN1_GP7[2][0]*UyN1 + dN2_GP7[2][0]*UyN2 + dN3_GP7[2][0]*UyN3 + dN4_GP7[2][0]*UyN4 + dN5_GP7[2][0]*UyN5 + dN6_GP7[2][0]*UyN6 + dN7_GP7[2][0]*UyN7 + dN8_GP7[2][0]*UyN8
        dUzdx_GP7= dN1_GP7[0][0]*UzN1 + dN2_GP7[0][0]*UzN2 + dN3_GP7[0][0]*UzN3 + dN4_GP7[0][0]*UzN4 + dN5_GP7[0][0]*UzN5 + dN6_GP7[0][0]*UzN6 + dN7_GP7[0][0]*UzN7 + dN8_GP7[0][0]*UzN8
        dUzdy_GP7= dN1_GP7[1][0]*UzN1 + dN2_GP7[1][0]*UzN2 + dN3_GP7[1][0]*UzN3 + dN4_GP7[1][0]*UzN4 + dN5_GP7[1][0]*UzN5 + dN6_GP7[1][0]*UzN6 + dN7_GP7[1][0]*UzN7 + dN8_GP7[1][0]*UzN8
        dUzdz_GP7= dN1_GP7[2][0]*UzN1 + dN2_GP7[2][0]*UzN2 + dN3_GP7[2][0]*UzN3 + dN4_GP7[2][0]*UzN4 + dN5_GP7[2][0]*UzN5 + dN6_GP7[2][0]*UzN6 + dN7_GP7[2][0]*UzN7 + dN8_GP7[2][0]*UzN8

        dUxdx_GP8= dN1_GP8[0][0]*UxN1 + dN2_GP8[0][0]*UxN2 + dN3_GP8[0][0]*UxN3 + dN4_GP8[0][0]*UxN4 + dN5_GP8[0][0]*UxN5 + dN6_GP8[0][0]*UxN6 + dN7_GP8[0][0]*UxN7 + dN8_GP8[0][0]*UxN8
        dUxdy_GP8= dN1_GP8[1][0]*UxN1 + dN2_GP8[1][0]*UxN2 + dN3_GP8[1][0]*UxN3 + dN4_GP8[1][0]*UxN4 + dN5_GP8[1][0]*UxN5 + dN6_GP8[1][0]*UxN6 + dN7_GP8[1][0]*UxN7 + dN8_GP8[1][0]*UxN8
        dUxdz_GP8= dN1_GP8[2][0]*UxN1 + dN2_GP8[2][0]*UxN2 + dN3_GP8[2][0]*UxN3 + dN4_GP8[2][0]*UxN4 + dN5_GP8[2][0]*UxN5 + dN6_GP8[2][0]*UxN6 + dN7_GP8[2][0]*UxN7 + dN8_GP8[2][0]*UxN8
        dUydx_GP8= dN1_GP8[0][0]*UyN1 + dN2_GP8[0][0]*UyN2 + dN3_GP8[0][0]*UyN3 + dN4_GP8[0][0]*UyN4 + dN5_GP8[0][0]*UyN5 + dN6_GP8[0][0]*UyN6 + dN7_GP8[0][0]*UyN7 + dN8_GP8[0][0]*UyN8
        dUydy_GP8= dN1_GP8[1][0]*UyN1 + dN2_GP8[1][0]*UyN2 + dN3_GP8[1][0]*UyN3 + dN4_GP8[1][0]*UyN4 + dN5_GP8[1][0]*UyN5 + dN6_GP8[1][0]*UyN6 + dN7_GP8[1][0]*UyN7 + dN8_GP8[1][0]*UyN8
        dUydz_GP8= dN1_GP8[2][0]*UyN1 + dN2_GP8[2][0]*UyN2 + dN3_GP8[2][0]*UyN3 + dN4_GP8[2][0]*UyN4 + dN5_GP8[2][0]*UyN5 + dN6_GP8[2][0]*UyN6 + dN7_GP8[2][0]*UyN7 + dN8_GP8[2][0]*UyN8
        dUzdx_GP8= dN1_GP8[0][0]*UzN1 + dN2_GP8[0][0]*UzN2 + dN3_GP8[0][0]*UzN3 + dN4_GP8[0][0]*UzN4 + dN5_GP8[0][0]*UzN5 + dN6_GP8[0][0]*UzN6 + dN7_GP8[0][0]*UzN7 + dN8_GP8[0][0]*UzN8
        dUzdy_GP8= dN1_GP8[1][0]*UzN1 + dN2_GP8[1][0]*UzN2 + dN3_GP8[1][0]*UzN3 + dN4_GP8[1][0]*UzN4 + dN5_GP8[1][0]*UzN5 + dN6_GP8[1][0]*UzN6 + dN7_GP8[1][0]*UzN7 + dN8_GP8[1][0]*UzN8
        dUzdz_GP8= dN1_GP8[2][0]*UzN1 + dN2_GP8[2][0]*UzN2 + dN3_GP8[2][0]*UzN3 + dN4_GP8[2][0]*UzN4 + dN5_GP8[2][0]*UzN5 + dN6_GP8[2][0]*UzN6 + dN7_GP8[2][0]*UzN7 + dN8_GP8[2][0]*UzN8

        # calculate stress, strain, strain energy and compliance for GP1
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_zz_GP1= dUzdz_GP1
        e_xy_GP1= 0.5 * (dUydx_GP1 + dUxdy_GP1)
        e_xz_GP1= 0.5 * (dUzdx_GP1 + dUxdz_GP1)
        e_yz_GP1= 0.5 * (dUydz_GP1 + dUzdy_GP1)
        e_xx_GP1_hat = e_xx_GP1.detach()
        e_yy_GP1_hat = e_yy_GP1.detach()
        e_zz_GP1_hat = e_zz_GP1.detach()
        e_xy_GP1_hat = e_xy_GP1.detach()
        e_xz_GP1_hat = e_xz_GP1.detach()
        e_yz_GP1_hat = e_yz_GP1.detach()

        S_xx_GP1_d = (lambdda + 2*mu) * e_xx_GP1 + lambdda * (e_yy_GP1 + e_zz_GP1)
        S_yy_GP1_d = (lambdda + 2*mu) * e_yy_GP1 + lambdda * (e_xx_GP1 + e_zz_GP1)
        S_zz_GP1_d = (lambdda + 2*mu) * e_zz_GP1 + lambdda * (e_xx_GP1 + e_yy_GP1)
        S_xy_GP1_d = 2*mu*e_xy_GP1
        S_xz_GP1_d = 2*mu*e_xz_GP1
        S_yz_GP1_d = 2*mu*e_yz_GP1
        S_xx_GP1_c = (lambdda_hat + 2*mu_hat) * e_xx_GP1_hat + lambdda_hat * (e_yy_GP1_hat + e_zz_GP1_hat)
        S_yy_GP1_c = (lambdda_hat + 2*mu_hat) * e_yy_GP1_hat + lambdda_hat * (e_xx_GP1_hat + e_zz_GP1_hat)
        S_zz_GP1_c = (lambdda_hat + 2*mu_hat) * e_zz_GP1_hat + lambdda_hat * (e_xx_GP1_hat + e_yy_GP1_hat)
        S_xy_GP1_c = 2*mu_hat*e_xy_GP1_hat
        S_xz_GP1_c = 2*mu_hat*e_xz_GP1_hat
        S_yz_GP1_c = 2*mu_hat*e_yz_GP1_hat

        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1_d + e_yy_GP1*S_yy_GP1_d + e_zz_GP1*S_zz_GP1_d 
                           + 2*e_xy_GP1*S_xy_GP1_d + 2*e_xz_GP1*S_xz_GP1_d + 2*e_yz_GP1*S_yz_GP1_d)
        compliance_GP1 = (e_xx_GP1_hat*S_xx_GP1_c + e_yy_GP1_hat*S_yy_GP1_c + e_zz_GP1_hat*S_zz_GP1_c
                           + 2*e_xy_GP1_hat*S_xy_GP1_c + 2*e_xz_GP1_hat*S_xz_GP1_c + 2*e_yz_GP1_hat*S_yz_GP1_c)
        
        # calculate stress, strain, strain energy and compliance for GP2
        e_xx_GP2= dUxdx_GP2
        e_yy_GP2= dUydy_GP2
        e_zz_GP2= dUzdz_GP2
        e_xy_GP2= 0.5 * (dUydx_GP2 + dUxdy_GP2)
        e_xz_GP2= 0.5 * (dUzdx_GP2 + dUxdz_GP2)
        e_yz_GP2= 0.5 * (dUydz_GP2 + dUzdy_GP2)
        e_xx_GP2_hat = e_xx_GP2.detach()
        e_yy_GP2_hat = e_yy_GP2.detach()
        e_zz_GP2_hat = e_zz_GP2.detach()
        e_xy_GP2_hat = e_xy_GP2.detach()
        e_xz_GP2_hat = e_xz_GP2.detach()
        e_yz_GP2_hat = e_yz_GP2.detach()

        S_xx_GP2_d = (lambdda + 2*mu) * e_xx_GP2 + lambdda * (e_yy_GP2 + e_zz_GP2)
        S_yy_GP2_d = (lambdda + 2*mu) * e_yy_GP2 + lambdda * (e_xx_GP2 + e_zz_GP2)
        S_zz_GP2_d = (lambdda + 2*mu) * e_zz_GP2 + lambdda * (e_xx_GP2 + e_yy_GP2)
        S_xy_GP2_d = 2*mu*e_xy_GP2
        S_xz_GP2_d = 2*mu*e_xz_GP2
        S_yz_GP2_d = 2*mu*e_yz_GP2
        S_xx_GP2_c = (lambdda_hat + 2*mu_hat) * e_xx_GP2_hat + lambdda_hat * (e_yy_GP2_hat + e_zz_GP2_hat)
        S_yy_GP2_c = (lambdda_hat + 2*mu_hat) * e_yy_GP2_hat + lambdda_hat * (e_xx_GP2_hat + e_zz_GP2_hat)
        S_zz_GP2_c = (lambdda_hat + 2*mu_hat) * e_zz_GP2_hat + lambdda_hat * (e_xx_GP2_hat + e_yy_GP2_hat)
        S_xy_GP2_c = 2*mu_hat*e_xy_GP2_hat
        S_xz_GP2_c = 2*mu_hat*e_xz_GP2_hat
        S_yz_GP2_c = 2*mu_hat*e_yz_GP2_hat

        strainEnergy_GP2= (e_xx_GP2*S_xx_GP2_d + e_yy_GP2*S_yy_GP2_d + e_zz_GP2*S_zz_GP2_d 
                           + 2*e_xy_GP2*S_xy_GP2_d + 2*e_xz_GP2*S_xz_GP2_d + 2*e_yz_GP2*S_yz_GP2_d)
        compliance_GP2 = (e_xx_GP2_hat*S_xx_GP2_c + e_yy_GP2_hat*S_yy_GP2_c + e_zz_GP2_hat*S_zz_GP2_c
                           + 2*e_xy_GP2_hat*S_xy_GP2_c + 2*e_xz_GP2_hat*S_xz_GP2_c + 2*e_yz_GP2_hat*S_yz_GP2_c)
        
        # calculate stress, strain, strain energy and compliance for GP3
        e_xx_GP3= dUxdx_GP3
        e_yy_GP3= dUydy_GP3
        e_zz_GP3= dUzdz_GP3
        e_xy_GP3= 0.5 * (dUydx_GP3 + dUxdy_GP3)
        e_xz_GP3= 0.5 * (dUzdx_GP3 + dUxdz_GP3)
        e_yz_GP3= 0.5 * (dUydz_GP3 + dUzdy_GP3)
        e_xx_GP3_hat = e_xx_GP3.detach()
        e_yy_GP3_hat = e_yy_GP3.detach()
        e_zz_GP3_hat = e_zz_GP3.detach()
        e_xy_GP3_hat = e_xy_GP3.detach()
        e_xz_GP3_hat = e_xz_GP3.detach()
        e_yz_GP3_hat = e_yz_GP3.detach()

        S_xx_GP3_d = (lambdda + 2*mu) * e_xx_GP3 + lambdda * (e_yy_GP3 + e_zz_GP3)
        S_yy_GP3_d = (lambdda + 2*mu) * e_yy_GP3 + lambdda * (e_xx_GP3 + e_zz_GP3)
        S_zz_GP3_d = (lambdda + 2*mu) * e_zz_GP3 + lambdda * (e_xx_GP3 + e_yy_GP3)
        S_xy_GP3_d = 2*mu*e_xy_GP3
        S_xz_GP3_d = 2*mu*e_xz_GP3
        S_yz_GP3_d = 2*mu*e_yz_GP3
        S_xx_GP3_c = (lambdda_hat + 2*mu_hat) * e_xx_GP3_hat + lambdda_hat * (e_yy_GP3_hat + e_zz_GP3_hat)
        S_yy_GP3_c = (lambdda_hat + 2*mu_hat) * e_yy_GP3_hat + lambdda_hat * (e_xx_GP3_hat + e_zz_GP3_hat)
        S_zz_GP3_c = (lambdda_hat + 2*mu_hat) * e_zz_GP3_hat + lambdda_hat * (e_xx_GP3_hat + e_yy_GP3_hat)
        S_xy_GP3_c = 2*mu_hat*e_xy_GP3_hat
        S_xz_GP3_c = 2*mu_hat*e_xz_GP3_hat
        S_yz_GP3_c = 2*mu_hat*e_yz_GP3_hat

        strainEnergy_GP3= (e_xx_GP3*S_xx_GP3_d + e_yy_GP3*S_yy_GP3_d + e_zz_GP3*S_zz_GP3_d 
                           + 2*e_xy_GP3*S_xy_GP3_d + 2*e_xz_GP3*S_xz_GP3_d + 2*e_yz_GP3*S_yz_GP3_d)
        compliance_GP3 = (e_xx_GP3_hat*S_xx_GP3_c + e_yy_GP3_hat*S_yy_GP3_c + e_zz_GP3_hat*S_zz_GP3_c
                           + 2*e_xy_GP3_hat*S_xy_GP3_c + 2*e_xz_GP3_hat*S_xz_GP3_c + 2*e_yz_GP3_hat*S_yz_GP3_c)
        
        # calculate stress, strain, strain energy and compliance for GP4
        e_xx_GP4= dUxdx_GP4
        e_yy_GP4= dUydy_GP4
        e_zz_GP4= dUzdz_GP4
        e_xy_GP4= 0.5 * (dUydx_GP4 + dUxdy_GP4)
        e_xz_GP4= 0.5 * (dUzdx_GP4 + dUxdz_GP4)
        e_yz_GP4= 0.5 * (dUydz_GP4 + dUzdy_GP4)
        e_xx_GP4_hat = e_xx_GP4.detach()
        e_yy_GP4_hat = e_yy_GP4.detach()
        e_zz_GP4_hat = e_zz_GP4.detach()
        e_xy_GP4_hat = e_xy_GP4.detach()
        e_xz_GP4_hat = e_xz_GP4.detach()
        e_yz_GP4_hat = e_yz_GP4.detach()

        S_xx_GP4_d = (lambdda + 2*mu) * e_xx_GP4 + lambdda * (e_yy_GP4 + e_zz_GP4)
        S_yy_GP4_d = (lambdda + 2*mu) * e_yy_GP4 + lambdda * (e_xx_GP4 + e_zz_GP4)
        S_zz_GP4_d = (lambdda + 2*mu) * e_zz_GP4 + lambdda * (e_xx_GP4 + e_yy_GP4)
        S_xy_GP4_d = 2*mu*e_xy_GP4
        S_xz_GP4_d = 2*mu*e_xz_GP4
        S_yz_GP4_d = 2*mu*e_yz_GP4
        S_xx_GP4_c = (lambdda_hat + 2*mu_hat) * e_xx_GP4_hat + lambdda_hat * (e_yy_GP4_hat + e_zz_GP4_hat)
        S_yy_GP4_c = (lambdda_hat + 2*mu_hat) * e_yy_GP4_hat + lambdda_hat * (e_xx_GP4_hat + e_zz_GP4_hat)
        S_zz_GP4_c = (lambdda_hat + 2*mu_hat) * e_zz_GP4_hat + lambdda_hat * (e_xx_GP4_hat + e_yy_GP4_hat)
        S_xy_GP4_c = 2*mu_hat*e_xy_GP4_hat
        S_xz_GP4_c = 2*mu_hat*e_xz_GP4_hat
        S_yz_GP4_c = 2*mu_hat*e_yz_GP4_hat

        strainEnergy_GP4= (e_xx_GP4*S_xx_GP4_d + e_yy_GP4*S_yy_GP4_d + e_zz_GP4*S_zz_GP4_d 
                           + 2*e_xy_GP4*S_xy_GP4_d + 2*e_xz_GP4*S_xz_GP4_d + 2*e_yz_GP4*S_yz_GP4_d)
        compliance_GP4 = (e_xx_GP4_hat*S_xx_GP4_c + e_yy_GP4_hat*S_yy_GP4_c + e_zz_GP4_hat*S_zz_GP4_c
                           + 2*e_xy_GP4_hat*S_xy_GP4_c + 2*e_xz_GP4_hat*S_xz_GP4_c + 2*e_yz_GP4_hat*S_yz_GP4_c)
        
        # calculate stress, strain, strain energy and compliance for GP5
        e_xx_GP5= dUxdx_GP5
        e_yy_GP5= dUydy_GP5
        e_zz_GP5= dUzdz_GP5
        e_xy_GP5= 0.5 * (dUydx_GP5 + dUxdy_GP5)
        e_xz_GP5= 0.5 * (dUzdx_GP5 + dUxdz_GP5)
        e_yz_GP5= 0.5 * (dUydz_GP5 + dUzdy_GP5)
        e_xx_GP5_hat = e_xx_GP5.detach()
        e_yy_GP5_hat = e_yy_GP5.detach()
        e_zz_GP5_hat = e_zz_GP5.detach()
        e_xy_GP5_hat = e_xy_GP5.detach()
        e_xz_GP5_hat = e_xz_GP5.detach()
        e_yz_GP5_hat = e_yz_GP5.detach()

        S_xx_GP5_d = (lambdda + 2*mu) * e_xx_GP5 + lambdda * (e_yy_GP5 + e_zz_GP5)
        S_yy_GP5_d = (lambdda + 2*mu) * e_yy_GP5 + lambdda * (e_xx_GP5 + e_zz_GP5)
        S_zz_GP5_d = (lambdda + 2*mu) * e_zz_GP5 + lambdda * (e_xx_GP5 + e_yy_GP5)
        S_xy_GP5_d = 2*mu*e_xy_GP5
        S_xz_GP5_d = 2*mu*e_xz_GP5
        S_yz_GP5_d = 2*mu*e_yz_GP5
        S_xx_GP5_c = (lambdda_hat + 2*mu_hat) * e_xx_GP5_hat + lambdda_hat * (e_yy_GP5_hat + e_zz_GP5_hat)
        S_yy_GP5_c = (lambdda_hat + 2*mu_hat) * e_yy_GP5_hat + lambdda_hat * (e_xx_GP5_hat + e_zz_GP5_hat)
        S_zz_GP5_c = (lambdda_hat + 2*mu_hat) * e_zz_GP5_hat + lambdda_hat * (e_xx_GP5_hat + e_yy_GP5_hat)
        S_xy_GP5_c = 2*mu_hat*e_xy_GP5_hat
        S_xz_GP5_c = 2*mu_hat*e_xz_GP5_hat
        S_yz_GP5_c = 2*mu_hat*e_yz_GP5_hat

        strainEnergy_GP5= (e_xx_GP5*S_xx_GP5_d + e_yy_GP5*S_yy_GP5_d + e_zz_GP5*S_zz_GP5_d 
                           + 2*e_xy_GP5*S_xy_GP5_d + 2*e_xz_GP5*S_xz_GP5_d + 2*e_yz_GP5*S_yz_GP5_d)
        compliance_GP5 = (e_xx_GP5_hat*S_xx_GP5_c + e_yy_GP5_hat*S_yy_GP5_c + e_zz_GP5_hat*S_zz_GP5_c
                           + 2*e_xy_GP5_hat*S_xy_GP5_c + 2*e_xz_GP5_hat*S_xz_GP5_c + 2*e_yz_GP5_hat*S_yz_GP5_c)
        
        # calculate stress, strain, strain energy and compliance for GP6
        e_xx_GP6= dUxdx_GP6
        e_yy_GP6= dUydy_GP6
        e_zz_GP6= dUzdz_GP6
        e_xy_GP6= 0.5 * (dUydx_GP6 + dUxdy_GP6)
        e_xz_GP6= 0.5 * (dUzdx_GP6 + dUxdz_GP6)
        e_yz_GP6= 0.5 * (dUydz_GP6 + dUzdy_GP6)
        e_xx_GP6_hat = e_xx_GP6.detach()
        e_yy_GP6_hat = e_yy_GP6.detach()
        e_zz_GP6_hat = e_zz_GP6.detach()
        e_xy_GP6_hat = e_xy_GP6.detach()
        e_xz_GP6_hat = e_xz_GP6.detach()
        e_yz_GP6_hat = e_yz_GP6.detach()

        S_xx_GP6_d = (lambdda + 2*mu) * e_xx_GP6 + lambdda * (e_yy_GP6 + e_zz_GP6)
        S_yy_GP6_d = (lambdda + 2*mu) * e_yy_GP6 + lambdda * (e_xx_GP6 + e_zz_GP6)
        S_zz_GP6_d = (lambdda + 2*mu) * e_zz_GP6 + lambdda * (e_xx_GP6 + e_yy_GP6)
        S_xy_GP6_d = 2*mu*e_xy_GP6
        S_xz_GP6_d = 2*mu*e_xz_GP6
        S_yz_GP6_d = 2*mu*e_yz_GP6
        S_xx_GP6_c = (lambdda_hat + 2*mu_hat) * e_xx_GP6_hat + lambdda_hat * (e_yy_GP6_hat + e_zz_GP6_hat)
        S_yy_GP6_c = (lambdda_hat + 2*mu_hat) * e_yy_GP6_hat + lambdda_hat * (e_xx_GP6_hat + e_zz_GP6_hat)
        S_zz_GP6_c = (lambdda_hat + 2*mu_hat) * e_zz_GP6_hat + lambdda_hat * (e_xx_GP6_hat + e_yy_GP6_hat)
        S_xy_GP6_c = 2*mu_hat*e_xy_GP6_hat
        S_xz_GP6_c = 2*mu_hat*e_xz_GP6_hat
        S_yz_GP6_c = 2*mu_hat*e_yz_GP6_hat

        strainEnergy_GP6= (e_xx_GP6*S_xx_GP6_d + e_yy_GP6*S_yy_GP6_d + e_zz_GP6*S_zz_GP6_d 
                           + 2*e_xy_GP6*S_xy_GP6_d + 2*e_xz_GP6*S_xz_GP6_d + 2*e_yz_GP6*S_yz_GP6_d)
        compliance_GP6 = (e_xx_GP6_hat*S_xx_GP6_c + e_yy_GP6_hat*S_yy_GP6_c + e_zz_GP6_hat*S_zz_GP6_c
                           + 2*e_xy_GP6_hat*S_xy_GP6_c + 2*e_xz_GP6_hat*S_xz_GP6_c + 2*e_yz_GP6_hat*S_yz_GP6_c)
        
        # calculate stress, strain, strain energy and compliance for GP7
        e_xx_GP7= dUxdx_GP7
        e_yy_GP7= dUydy_GP7
        e_zz_GP7= dUzdz_GP7
        e_xy_GP7= 0.5 * (dUydx_GP7 + dUxdy_GP7)
        e_xz_GP7= 0.5 * (dUzdx_GP7 + dUxdz_GP7)
        e_yz_GP7= 0.5 * (dUydz_GP7 + dUzdy_GP7)
        e_xx_GP7_hat = e_xx_GP7.detach()
        e_yy_GP7_hat = e_yy_GP7.detach()
        e_zz_GP7_hat = e_zz_GP7.detach()
        e_xy_GP7_hat = e_xy_GP7.detach()
        e_xz_GP7_hat = e_xz_GP7.detach()
        e_yz_GP7_hat = e_yz_GP7.detach()

        S_xx_GP7_d = (lambdda + 2*mu) * e_xx_GP7 + lambdda * (e_yy_GP7 + e_zz_GP7)
        S_yy_GP7_d = (lambdda + 2*mu) * e_yy_GP7 + lambdda * (e_xx_GP7 + e_zz_GP7)
        S_zz_GP7_d = (lambdda + 2*mu) * e_zz_GP7 + lambdda * (e_xx_GP7 + e_yy_GP7)
        S_xy_GP7_d = 2*mu*e_xy_GP7
        S_xz_GP7_d = 2*mu*e_xz_GP7
        S_yz_GP7_d = 2*mu*e_yz_GP7
        S_xx_GP7_c = (lambdda_hat + 2*mu_hat) * e_xx_GP7_hat + lambdda_hat * (e_yy_GP7_hat + e_zz_GP7_hat)
        S_yy_GP7_c = (lambdda_hat + 2*mu_hat) * e_yy_GP7_hat + lambdda_hat * (e_xx_GP7_hat + e_zz_GP7_hat)
        S_zz_GP7_c = (lambdda_hat + 2*mu_hat) * e_zz_GP7_hat + lambdda_hat * (e_xx_GP7_hat + e_yy_GP7_hat)
        S_xy_GP7_c = 2*mu_hat*e_xy_GP7_hat
        S_xz_GP7_c = 2*mu_hat*e_xz_GP7_hat
        S_yz_GP7_c = 2*mu_hat*e_yz_GP7_hat

        strainEnergy_GP7= (e_xx_GP7*S_xx_GP7_d + e_yy_GP7*S_yy_GP7_d + e_zz_GP7*S_zz_GP7_d 
                           + 2*e_xy_GP7*S_xy_GP7_d + 2*e_xz_GP7*S_xz_GP7_d + 2*e_yz_GP7*S_yz_GP7_d)
        compliance_GP7 = (e_xx_GP7_hat*S_xx_GP7_c + e_yy_GP7_hat*S_yy_GP7_c + e_zz_GP7_hat*S_zz_GP7_c
                           + 2*e_xy_GP7_hat*S_xy_GP7_c + 2*e_xz_GP7_hat*S_xz_GP7_c + 2*e_yz_GP7_hat*S_yz_GP7_c)
        
        # calculate stress, strain, strain energy and compliance for GP8
        e_xx_GP8= dUxdx_GP8
        e_yy_GP8= dUydy_GP8
        e_zz_GP8= dUzdz_GP8
        e_xy_GP8= 0.5 * (dUydx_GP8 + dUxdy_GP8)
        e_xz_GP8= 0.5 * (dUzdx_GP8 + dUxdz_GP8)
        e_yz_GP8= 0.5 * (dUydz_GP8 + dUzdy_GP8)
        e_xx_GP8_hat = e_xx_GP8.detach()
        e_yy_GP8_hat = e_yy_GP8.detach()
        e_zz_GP8_hat = e_zz_GP8.detach()
        e_xy_GP8_hat = e_xy_GP8.detach()
        e_xz_GP8_hat = e_xz_GP8.detach()
        e_yz_GP8_hat = e_yz_GP8.detach()

        S_xx_GP8_d = (lambdda + 2*mu) * e_xx_GP8 + lambdda * (e_yy_GP8 + e_zz_GP8)
        S_yy_GP8_d = (lambdda + 2*mu) * e_yy_GP8 + lambdda * (e_xx_GP8 + e_zz_GP8)
        S_zz_GP8_d = (lambdda + 2*mu) * e_zz_GP8 + lambdda * (e_xx_GP8 + e_yy_GP8)
        S_xy_GP8_d = 2*mu*e_xy_GP8
        S_xz_GP8_d = 2*mu*e_xz_GP8
        S_yz_GP8_d = 2*mu*e_yz_GP8
        S_xx_GP8_c = (lambdda_hat + 2*mu_hat) * e_xx_GP8_hat + lambdda_hat * (e_yy_GP8_hat + e_zz_GP8_hat)
        S_yy_GP8_c = (lambdda_hat + 2*mu_hat) * e_yy_GP8_hat + lambdda_hat * (e_xx_GP8_hat + e_zz_GP8_hat)
        S_zz_GP8_c = (lambdda_hat + 2*mu_hat) * e_zz_GP8_hat + lambdda_hat * (e_xx_GP8_hat + e_yy_GP8_hat)
        S_xy_GP8_c = 2*mu_hat*e_xy_GP8_hat
        S_xz_GP8_c = 2*mu_hat*e_xz_GP8_hat
        S_yz_GP8_c = 2*mu_hat*e_yz_GP8_hat

        strainEnergy_GP8= (e_xx_GP8*S_xx_GP8_d + e_yy_GP8*S_yy_GP8_d + e_zz_GP8*S_zz_GP8_d 
                           + 2*e_xy_GP8*S_xy_GP8_d + 2*e_xz_GP8*S_xz_GP8_d + 2*e_yz_GP8*S_yz_GP8_d)
        compliance_GP8 = (e_xx_GP8_hat*S_xx_GP8_c + e_yy_GP8_hat*S_yy_GP8_c + e_zz_GP8_hat*S_zz_GP8_c
                           + 2*e_xy_GP8_hat*S_xy_GP8_c + 2*e_xz_GP8_hat*S_xz_GP8_c + 2*e_yz_GP8_hat*S_yz_GP8_c)
        
        SE = (( strainEnergy_GP1 + strainEnergy_GP2 + strainEnergy_GP3 + strainEnergy_GP4 + strainEnergy_GP5 + strainEnergy_GP6 + strainEnergy_GP7 + strainEnergy_GP8) * detJ * 0.5).flatten()
        compliance = (( compliance_GP1 + compliance_GP2 + compliance_GP3 + compliance_GP4 + compliance_GP5 + compliance_GP6 + compliance_GP7 + compliance_GP8 ) * detJ).flatten()
        SE_masked = SE[mask_col_elem]
        compliance_masked = compliance[mask_col_elem]
        strain_energy = torch.sum( SE_masked )
        loss_compliance = torch.sum( compliance_masked )

    elif Diff_type == 'hex_reduced':
        # dN = [dN1_GP1, dN2_GP1, dN3_GP1, dN4_GP1, dN5_GP1, dN6_GP1, dN7_GP1, dN8_GP1]
        detJ = model_list[0].detJ
        dN = model_list[0].dN
        dN1_GP1 = dN[0]
        dN2_GP1 = dN[1]
        dN3_GP1 = dN[2]
        dN4_GP1 = dN[3]
        dN5_GP1 = dN[4]
        dN6_GP1 = dN[5]
        dN7_GP1 = dN[6]
        dN8_GP1 = dN[7]

        Ux= u.reshape(Nx, Ny, Nz)
        Uy= v.reshape(Nx, Ny, Nz)
        Uz= w.reshape(Nx, Ny, Nz)

        UxN1= Ux[:-1,:-1,:-1]
        UxN2= Ux[1:,:-1,:-1]
        UxN3= Ux[1:,1:,:-1]
        UxN4= Ux[:-1,1:,:-1]
        UxN5= Ux[:-1,:-1,1:]
        UxN6= Ux[1:,:-1,1:]
        UxN7= Ux[1:,1:,1:]
        UxN8= Ux[:-1,1:,1:]

        UyN1= Uy[:-1,:-1,:-1]
        UyN2= Uy[1:,:-1,:-1]
        UyN3= Uy[1:,1:,:-1]
        UyN4= Uy[:-1,1:,:-1]
        UyN5= Uy[:-1,:-1,1:]
        UyN6= Uy[1:,:-1,1:]
        UyN7= Uy[1:,1:,1:]
        UyN8= Uy[:-1,1:,1:]

        UzN1= Uz[:-1,:-1,:-1]
        UzN2= Uz[1:,:-1,:-1]
        UzN3= Uz[1:,1:,:-1]
        UzN4= Uz[:-1,1:,:-1]
        UzN5= Uz[:-1,:-1,1:]
        UzN6= Uz[1:,:-1,1:]
        UzN7= Uz[1:,1:,1:]
        UzN8= Uz[:-1,1:,1:]

        dUxdx_GP1= dN1_GP1[0][0]*UxN1 + dN2_GP1[0][0]*UxN2 + dN3_GP1[0][0]*UxN3 + dN4_GP1[0][0]*UxN4 + dN5_GP1[0][0]*UxN5 + dN6_GP1[0][0]*UxN6 + dN7_GP1[0][0]*UxN7 + dN8_GP1[0][0]*UxN8
        dUxdy_GP1= dN1_GP1[1][0]*UxN1 + dN2_GP1[1][0]*UxN2 + dN3_GP1[1][0]*UxN3 + dN4_GP1[1][0]*UxN4 + dN5_GP1[1][0]*UxN5 + dN6_GP1[1][0]*UxN6 + dN7_GP1[1][0]*UxN7 + dN8_GP1[1][0]*UxN8
        dUxdz_GP1= dN1_GP1[2][0]*UxN1 + dN2_GP1[2][0]*UxN2 + dN3_GP1[2][0]*UxN3 + dN4_GP1[2][0]*UxN4 + dN5_GP1[2][0]*UxN5 + dN6_GP1[2][0]*UxN6 + dN7_GP1[2][0]*UxN7 + dN8_GP1[2][0]*UxN8
        dUydx_GP1= dN1_GP1[0][0]*UyN1 + dN2_GP1[0][0]*UyN2 + dN3_GP1[0][0]*UyN3 + dN4_GP1[0][0]*UyN4 + dN5_GP1[0][0]*UyN5 + dN6_GP1[0][0]*UyN6 + dN7_GP1[0][0]*UyN7 + dN8_GP1[0][0]*UyN8
        dUydy_GP1= dN1_GP1[1][0]*UyN1 + dN2_GP1[1][0]*UyN2 + dN3_GP1[1][0]*UyN3 + dN4_GP1[1][0]*UyN4 + dN5_GP1[1][0]*UyN5 + dN6_GP1[1][0]*UyN6 + dN7_GP1[1][0]*UyN7 + dN8_GP1[1][0]*UyN8
        dUydz_GP1= dN1_GP1[2][0]*UyN1 + dN2_GP1[2][0]*UyN2 + dN3_GP1[2][0]*UyN3 + dN4_GP1[2][0]*UyN4 + dN5_GP1[2][0]*UyN5 + dN6_GP1[2][0]*UyN6 + dN7_GP1[2][0]*UyN7 + dN8_GP1[2][0]*UyN8
        dUzdx_GP1= dN1_GP1[0][0]*UzN1 + dN2_GP1[0][0]*UzN2 + dN3_GP1[0][0]*UzN3 + dN4_GP1[0][0]*UzN4 + dN5_GP1[0][0]*UzN5 + dN6_GP1[0][0]*UzN6 + dN7_GP1[0][0]*UzN7 + dN8_GP1[0][0]*UzN8
        dUzdy_GP1= dN1_GP1[1][0]*UzN1 + dN2_GP1[1][0]*UzN2 + dN3_GP1[1][0]*UzN3 + dN4_GP1[1][0]*UzN4 + dN5_GP1[1][0]*UzN5 + dN6_GP1[1][0]*UzN6 + dN7_GP1[1][0]*UzN7 + dN8_GP1[1][0]*UzN8
        dUzdz_GP1= dN1_GP1[2][0]*UzN1 + dN2_GP1[2][0]*UzN2 + dN3_GP1[2][0]*UzN3 + dN4_GP1[2][0]*UzN4 + dN5_GP1[2][0]*UzN5 + dN6_GP1[2][0]*UzN6 + dN7_GP1[2][0]*UzN7 + dN8_GP1[2][0]*UzN8

        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_zz_GP1= dUzdz_GP1
        e_xy_GP1= 0.5 * (dUydx_GP1 + dUxdy_GP1)
        e_xz_GP1= 0.5 * (dUzdx_GP1 + dUxdz_GP1)
        e_yz_GP1= 0.5 * (dUydz_GP1 + dUzdy_GP1)
        e_xx_GP1_hat = e_xx_GP1.detach()
        e_yy_GP1_hat = e_yy_GP1.detach()
        e_zz_GP1_hat = e_zz_GP1.detach()
        e_xy_GP1_hat = e_xy_GP1.detach()
        e_xz_GP1_hat = e_xz_GP1.detach()
        e_yz_GP1_hat = e_yz_GP1.detach()

        S_xx_GP1_d = (lambdda + 2*mu) * e_xx_GP1 + lambdda * (e_yy_GP1 + e_zz_GP1)
        S_yy_GP1_d = (lambdda + 2*mu) * e_yy_GP1 + lambdda * (e_xx_GP1 + e_zz_GP1)
        S_zz_GP1_d = (lambdda + 2*mu) * e_zz_GP1 + lambdda * (e_xx_GP1 + e_yy_GP1)
        S_xy_GP1_d = 2*mu*e_xy_GP1
        S_xz_GP1_d = 2*mu*e_xz_GP1
        S_yz_GP1_d = 2*mu*e_yz_GP1
        S_xx_GP1_c = (lambdda_hat + 2*mu_hat) * e_xx_GP1_hat + lambdda_hat * (e_yy_GP1_hat + e_zz_GP1_hat)
        S_yy_GP1_c = (lambdda_hat + 2*mu_hat) * e_yy_GP1_hat + lambdda_hat * (e_xx_GP1_hat + e_zz_GP1_hat)
        S_zz_GP1_c = (lambdda_hat + 2*mu_hat) * e_zz_GP1_hat + lambdda_hat * (e_xx_GP1_hat + e_yy_GP1_hat)
        S_xy_GP1_c = 2*mu_hat*e_xy_GP1_hat
        S_xz_GP1_c = 2*mu_hat*e_xz_GP1_hat
        S_yz_GP1_c = 2*mu_hat*e_yz_GP1_hat

        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1_d + e_yy_GP1*S_yy_GP1_d + e_zz_GP1*S_zz_GP1_d 
                           + 2*e_xy_GP1*S_xy_GP1_d + 2*e_xz_GP1*S_xz_GP1_d + 2*e_yz_GP1*S_yz_GP1_d)
        compliance_GP1 = (e_xx_GP1_hat*S_xx_GP1_c + e_yy_GP1_hat*S_yy_GP1_c + e_zz_GP1_hat*S_zz_GP1_c
                           + 2*e_xy_GP1_hat*S_xy_GP1_c + 2*e_xz_GP1_hat*S_xz_GP1_c + 2*e_yz_GP1_hat*S_yz_GP1_c)
        
        SE = (8.0 * ( strainEnergy_GP1 ) * detJ * 0.5).flatten()
        compliance = (8.0 * ( compliance_GP1 ) * detJ).flatten()
        SE_masked = SE[mask_col_elem]
        compliance_masked = compliance[mask_col_elem]
        strain_energy = torch.sum( SE_masked )
        loss_compliance = torch.sum( compliance_masked )
    else: # ND2
        pass

    return strain_energy, external_work,loss_compliance,loss_mConstraint,massfrac,loss_cConstraint,costfrac,grey_fraction

############################### Define Parameters ##############################################
# define material properties and
D = [0, 1.0]  # must be the ascend order
E = [1e-5, 1.0] 
P = [0, 1.0] # cost
nu = 0.3 # Poisson's ratio
p0 = 3.0 # initial penalty
pf = 3.0 # final penalty
massfrac_f = 0.1 # final mass fraction
costfrac_f = 0.2 # final mass fraction
frac_decrease = 0.5 # fraction of epoch to decrease volume fraction from VF0 to VF_f
thres_static = 0.5 # static threshold values to binarize the density field
rho_min = 0.1 # lower limit to define grey element (rho_min,rho_max)
rho_max = 0.9 # upper limit to define grey element (rho_min,rho_max)

#define structured elements
pad = 0 # padding layer thickness
Nelx = 60 # number of elements along x to define training dataset
Nely = 30 # number of elements along y to define training dataset
Nelz = 20 # number of elements along y to define training dataset

Nelx_max = 91
Nely_max = 46
Nelz_max = 31
Nelx_min = 60
Nely_min = 30
Nelz_min = 20
num_CP = 100
xmin = 0.0
xmax = 60.0
ymin = 0.0
ymax = 30.0
zmin = 0.0
zmax = 20.0
hole_center = None
hole_radius = None
ring_thickness = None

# define model parameters and training parameters
random_state = [1,3,5,7,9,11,13,15,17,19]
Example = 'EX3D2'
Case = 'single'
dynamic_weight = False
gradient_clip = False
Diff_type = 'hex_reduced' 
omega = 0.5
learning_rate_disp = 1e-3
learning_rate_rho = 1e-4
nrmThreshold = 0.1; # maximum allowable norm of the gradients, [-nrmThreshold,nrmThreshold]
TO_num_iter = 10000 # number of total topology optimization iteration
plot_num = 10 # number of plots to save
delta = 1e-1
wc = 1 # weight factor for loss_compliance
wd = 1e3 # weight factor for loss_dem
wm = 1e3 # weight factor for mass constraint 
wp = 1e3 # weight factor for cost constraint  
basis = 'PGCAN3D'#'PGCAN'#'neural_network' #'M3'  
basis_rho = 'PGCAN3D'
quant_correlation_class = 'Rough_RBF' 
activation = 'tanh'
if basis == 'PGCAN3D':
    n_features = 128 # must be a even number, only used for PGCAN
    n_cells = 3 # no more than 4, only used for PGCAN
    res = [22,11,8] # only for PGCAN3D [x,y,z]
    n_neurons = int(n_features/2)
    n_layers = 3 # 3 layers at most for PGCAN
    NN_arch = [n_neurons] * n_layers
    groups = n_features
    kernel_size = [4,2,1]
else:
    n_features = None #only used for PGCAN
    n_cells = None #only used for PGCAN
    res = None # only for PGCAN
    n_neurons = 64
    n_layers = 6
    NN_arch = [n_neurons] * n_layers 
    groups = None
    kernel_size = None

############################### Pre-processing ##############################################
num_phase = len(D)
# Convert D to a tensor and reshape for broadcasting
D_tensor = torch.tensor(D, dtype=tkwargs["dtype"], device=tkwargs['device'])  # shape: [n_phases]
E_tensor = torch.tensor(E, dtype=tkwargs["dtype"], device=tkwargs['device'])  # shape: [n_phases]
P_tensor = torch.tensor(P, dtype=tkwargs["dtype"], device=tkwargs['device'])  # shape: [n_phases]

frac_decrease = int(frac_decrease * TO_num_iter)
plotting_interval_TO = TO_num_iter/plot_num
frac_step = None
plotting_interval_TO = TO_num_iter/plot_num
domain = {'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax]}
MP = {'hole_center':hole_center,'hole_radius':hole_radius,'ring_thickness':ring_thickness,'pad':pad,
      'Nelx': Nelx, 'Nely': Nely,'Nelz': Nelz,'Nelx_max': Nelx_max, 'Nely_max': Nely_max, 'Nelz_max': Nelz_max, 
      'Nelx_min': Nelx_min, 'Nely_min': Nely_min, 'Nelz_min': Nelz_min, 'num_CP':num_CP,'num_phase':num_phase,
      'domain': domain,'thres_static':thres_static,'Diff_type':Diff_type,
      'D':D_tensor,'E': E_tensor,'P':P_tensor, 'nu': nu,'p':p0,'pf':pf,'p0':p0,'rho_min':rho_min,'rho_max':rho_max,
      'massfrac_star':massfrac_f, 'massfrac0':massfrac_f, 'massfrac_f':massfrac_f,'frac_decrease':frac_decrease,
      'costfrac_star':costfrac_f, 'costfrac0':costfrac_f, 'costfrac_f':costfrac_f,
      'frac_step_mass':frac_step,'frac_step_cost':frac_step,'frac_step_p':frac_step}

NN_config_disp = {'random_state':random_state,'state':[],'Example':Example,'Case':Case,
             'dynamic_weight':dynamic_weight,'gradient_clip':gradient_clip,
             'omega':omega,'learning_rate_disp':learning_rate_disp,'learning_rate_rho':learning_rate_rho,'nrmThreshold':nrmThreshold,
             'TO_num_iter':TO_num_iter,'plot_num':plot_num,'plotting_interval_TO':plotting_interval_TO,
             'delta':delta,'wc':wc,'wd':wd,'wm':wm,'wp':wp,
             'basis': basis, 'quant_correlation_class': quant_correlation_class, 'activation': activation,
             'n_features': n_features, 'n_cells': n_cells, 'res': res,'groups':groups,'kernel_size':kernel_size,
             'NN_arch': NN_arch, 'save_folder': []}
# construct NN_config for the density network
NN_config_rho = NN_config_disp
NN_config_rho['basis'] = basis_rho
if basis_rho == 'PGCAN3D':
    NN_config_rho['n_features'] = 128 # must be a even number, only used for PGCAN
    NN_config_rho['n_cells'] = 6 # no more than 4, only used for PGCAN
    NN_config_rho['res'] = [22,11,8] # only for PGCAN3D [x,y,z]
    n_neurons = int(NN_config_rho['n_features']/2)
    n_layers = 3 # 3 layers at most for PGCAN
    NN_config_rho['NN_arch'] = [n_neurons] * n_layers
    NN_config_rho['groups'] = NN_config_rho['n_features']
    NN_config_rho['kernel_size'] = [4,2,1]
else:
    NN_config_rho['n_features'] = None #only used for PGCAN
    NN_config_rho['n_cells'] = None #only used for PGCAN
    NN_config_rho['res'] = None # only for PGCAN
    n_neurons = 64
    n_layers = 6
    NN_config_rho['NN_arch'] = [n_neurons] * n_layers 
    NN_config_rho['groups'] = None
    NN_config_rho['kernel_size'] = None

############################### Generate Data ##############################################
Training, X_col_all = get_data(MP)
X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
w_X_train = Training['w_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
w_train = Training['w_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

# Generate random integers
index_CP = np.random.randint(0, num_CP, size=TO_num_iter)

############################### TO Loop ##########################################
base_folder = f"/home/alexsunuci/PIGP3D_Aug12th_2025/Results/{Example}/{Case}/"
# base_folder = f"Results/{Example}/{Case}/"

# Create folders for each random state
for i, state in enumerate(random_state, start=1):
    
    # set the random seed number:
    set_seed0(state)
    NN_config_disp['state'] = state

    # create the folder
    save_folder = f"{base_folder}Run_{i}/"
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    NN_config_disp['save_folder'] = save_folder

    # define the models: these models shared the same NN (using model_u)
    model_u = LMGP(train_x = u_X_train, train_y = u_train, collocation_x = X_col,
                    NN_config = NN_config_disp,
                    Training = Training,
                    name_output='u',
                    MP=MP,
                    num_output = 3).to(tkwargs['device'])

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                    NN_config = NN_config_disp,
                    Training = Training,
                    name_output='v',
                    MP=MP,
                    num_output = 3).to(tkwargs['device'])

    model_w = LMGP(train_x = w_X_train, train_y = w_train, collocation_x = X_col,
                    NN_config = NN_config_disp,
                    Training = Training,
                    name_output='w',
                    MP=MP,
                    num_output = 3).to(tkwargs['device'])

    model_phase = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                    NN_config = NN_config_rho,
                    Training = Training,
                    name_output='rho',
                    MP=MP,
                    num_output = num_phase).to(tkwargs['device'])

    model_list = [model_u, model_v, model_w, model_phase]

    # define the time history dict
    timeHistory = {'loss_total':[],'loss_compliance':[],'loss_dem':[],'loss_mConstraint':[],'loss_cConstraint':[],'grey':[],
               'strain_energy':[],'external_work':[],'massfrac':[],'costfrac':[],'wc':[], 'wd':[], 'wm':[],'wp':[],'p':[]}  
    
    # Define optimizer for just the two models
    optimizer_disp = torch.optim.Adam(model_list[0].parameters(),lr=learning_rate_disp,amsgrad=True)
    optimizer_rho = torch.optim.Adam(model_list[3].parameters(),lr=learning_rate_rho,amsgrad=True)
    
    # Learning rate scheduler
    scheduler_disp = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disp, 
            milestones=np.linspace(0, TO_num_iter, 4).tolist(), 
            gamma=0.75)
    scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_rho, 
            milestones=np.linspace(0, TO_num_iter, 4).tolist(), 
            gamma=0.75)

    # Training loop
    epochs_iter = tqdm(range(TO_num_iter), desc='TO Epoch', position=0, leave=True)
    total_time = 0
    start_time = time.time()  # get the current time
    for epoch in epochs_iter:
        print(f"Starting epoch {epoch}")
        # zero gradients from previous iteration
        optimizer_disp.zero_grad()
        optimizer_rho.zero_grad()

        # update parameters and collocation points
        index = index_CP[epoch]
        model_list[0].dx = X_col_all[index]['dx']
        model_list[0].dy = X_col_all[index]['dy']
        model_list[0].dz = X_col_all[index]['dz']
        model_list[0].Nx = X_col_all[index]['Nx']
        model_list[0].Ny = X_col_all[index]['Ny']
        model_list[0].Nz = X_col_all[index]['Nz']
        model_list[0].mask_col = X_col_all[index]['mask_col']
        model_list[0].collocation_x = X_col_all[index]['X_col']
        model_list[0].mask_col_elem = X_col_all[index]['mask_col_elem']
        model_list[0].elem_x = X_col_all[index]['X_col_elem']
        model_list[0].detJ = X_col_all[index]['detJ']
        model_list[0].dN = X_col_all[index]['dN']
        model_list[0].traction_indices = X_col_all[index]['traction_indices']
        model_list[0].traction_magnitude = X_col_all[index]['traction_magnitude']
        
        if epoch == 0:
            initial_massfrac = eval_initial_massfrac(model_list)
            initial_costfrac = eval_initial_costfrac(model_list)
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
            
        # calculate DEM loss: (detach rho_element, used to monior loss_compliance_fem)
        strain_energy, external_work,loss_compliance,loss_mConstraint,massfrac,loss_cConstraint,costfrac,grey_fraction = calculate_TO_loss(model_list,Diff_type)

        offset = (1 + delta)/2 * external_work.detach()
        loss_dem = (strain_energy - external_work + offset)

        loss =  wc * loss_compliance + wd * loss_dem + wm * loss_mConstraint + wp * loss_cConstraint
        
        loss.backward(retain_graph=True)
        # Clip gradients to avoid NaN or inf values
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
            torch.nn.utils.clip_grad_norm_(model_list[3].parameters(), nrmThreshold)

        optimizer_disp.step()
        optimizer_rho.step()
        scheduler_disp.step()
        scheduler_rho.step()
        
        # save the loss histories and plot
        timeHistory['loss_total'].append(loss.item())
        timeHistory['loss_compliance'].append(loss_compliance.item())
        timeHistory['loss_dem'].append(loss_dem.item())
        timeHistory['loss_mConstraint'].append(loss_mConstraint.item())
        timeHistory['loss_cConstraint'].append(loss_cConstraint.item())
        timeHistory['massfrac'].append(massfrac.item())
        timeHistory['costfrac'].append(costfrac.item())
        timeHistory['strain_energy'].append(strain_energy.item())
        timeHistory['external_work'].append(external_work.item())
        timeHistory['wc'].append(wc)
        timeHistory['wd'].append(wd)
        timeHistory['wm'].append(wm)
        timeHistory['wp'].append(wp)
        timeHistory['grey'].append(grey_fraction)
        timeHistory['p'].append(model_list[0].MP['p'])
        
        # visualize contours and history
        if  (epoch+1) % plotting_interval_TO == 0:
            end_time = time.time()
            # save the model_list[0] to dict
            gpu_id = 0  # Assuming you are using GPU 0
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            # torch.save(model_list[0].mean_module_NN_All.state_dict(),save_folder + f'Trained_mean_module_NN_params_{epoch}.pth')
            torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[3].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_{epoch}.pth')
            total_time = total_time + (end_time - start_time)
            start_time = time.time()

        model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
        model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
        model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])

    end_time = time.time()
    total_time = total_time + (end_time - start_time)

    # save the MP to a pt file
    torch.save(MP, save_folder + "MP.pt")  # binary format
    
    # save the NN_config to a JSON file
    with open(save_folder + "NN_config_disp.json", "w") as file:
        json.dump(NN_config_disp, file, indent=4)
    with open(save_folder + "NN_config_rho.json", "w") as file:
        json.dump(NN_config_rho, file, indent=4)

    # save the Training to file
    torch.save(Training, save_folder + "Training.pth")

    # save the model_list[0] to dict
    torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[3].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_final.pth')

    # save the time history
    with open(save_folder + "timeHistory.json", "w") as file:
        json.dump(timeHistory, file)

    # Write the loss terms to the file
    result_file_path = f"{save_folder}Results_summary.txt"
    with open(result_file_path, 'w') as f:
        f.write(f"The total training time in second is: {total_time}\n")
        f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
        f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")
        f.write(f"Final value Loss_total: {timeHistory['loss_total'][-1]}\n\n")


