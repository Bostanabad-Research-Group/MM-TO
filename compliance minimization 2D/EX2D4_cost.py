import os
import json
import torch
import numpy as np
import time
import shutil
from models.lmgp import LMGP 
from gpytorch.settings import cholesky_jitter
from tqdm import tqdm
from utils.utils_general import set_seed0, get_tkwargs
from utils.utils_general import compute_dynamic_weight_2 as compute_dynamic_weight
from utils.get_training_data_2D import get_data_EX2D4 as get_data

tkwargs = get_tkwargs()

def eval_initial_massfrac(model_list):
    D = model_list[0].MP['D']  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    elem_volume = dx * dy
    
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()
    
    for model in model_list:
        model.eval()
    
    # Output raw phase logits
    m_phase = model_list[2].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
    # Apply softmax across phase dimension (dim=1)
    phase_weights = m_phase
    
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
    elem_volume = dx * dy
    
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()
    
    for model in model_list:
        model.eval()
    
    # Output raw phase logits
    m_phase = model_list[2].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
    # Apply softmax across phase dimension (dim=1)
    phase_weights = m_phase

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
    dx = model_list[0].dx
    dy = model_list[0].dy
    elem_volume = dx * dy
    E_tensor = model_list[0].MP['E'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor

    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    massfrac_star = model_list[0].MP['massfrac_star']
    costfrac_star = model_list[0].MP['costfrac_star']
    Nx = model_list[0].Nx
    Ny = model_list[0].Ny
    # mask_col = model_list[0].mask_col
    collocation_x = model_list[0].collocation_x.clone()
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()

    for model in model_list:
        model.train

    m_col = model_list[0].mean_module_NN_All(collocation_x)
    m_phase = model_list[2].mean_module_NN_All(elem_x)

    g_u = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()
    g_v = (model_list[1].covar_module(model_list[1].train_inputs[0], collocation_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()
            model_list[1].chol_decomp = model_list[1].covar_module(model_list[1].train_inputs[0]).cholesky()

    K_inv_offset_u = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))
    K_inv_offset_v = model_list[1].chol_decomp._cholesky_solve(model_list[1].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[1].train_inputs[0])[:,1].unsqueeze(-1))

    u = (m_col[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u)#.squeeze(-1)
    v = (m_col[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v)#.squeeze(-1)

    # Apply softmax across phase dimension (dim=1)
    phase_weights = m_phase
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
    E_tensor = model_list[0].MP['E'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor on the same device
    phase_weights_no_grad = phase_weights.detach() # shape: [n_elem, n_phases]
    
    E_node = (E_tensor * phase_weights_no_grad ** p).sum(dim = 1) # shape: [n_elem]
    E_hat_node = (E_tensor * (phase_weights_no_grad)**(2*p)/((phase_weights + 1e-8) ** p)).sum(dim = 1) # shape: [n_elem]

    E = E_node.reshape(Nx-1, Ny-1).T.flip(dims=[0])
    E_hat = E_hat_node.reshape(Nx-1, Ny-1).T.flip(dims=[0])

    # use central difference method
    if Diff_type == 'AD': #autograd
        pass
    elif Diff_type == 'quad4':
        detJ = model_list[0].detJ
        dN = model_list[0].dN
        dN1_dxy_GP1 = dN[0]
        dN1_dxy_GP2 = dN[1]
        dN1_dxy_GP3 = dN[2]
        dN1_dxy_GP4 = dN[3]
        dN2_dxy_GP1 = dN[4]
        dN2_dxy_GP2 = dN[5]
        dN2_dxy_GP3 = dN[6]
        dN2_dxy_GP4 = dN[7]
        dN3_dxy_GP1 = dN[8]
        dN3_dxy_GP2 = dN[9]
        dN3_dxy_GP3 = dN[10]
        dN3_dxy_GP4 = dN[11]
        dN4_dxy_GP1 = dN[12]
        dN4_dxy_GP2 = dN[13]
        dN4_dxy_GP3 = dN[14]
        dN4_dxy_GP4 = dN[15]

        Ux= u.reshape(Nx, Ny).T.flip(dims=[0])
        Uy= v.reshape(Nx, Ny).T.flip(dims=[0])
        
        UxN1= Ux[1:, :-1]
        UxN2= Ux[1:, 1:]
        UxN3= Ux[:-1, 1:]
        UxN4= Ux[:-1, :-1]

        UyN1= Uy[1:, :-1]
        UyN2= Uy[1:, 1:]
        UyN3= Uy[:-1, 1:]
        UyN4= Uy[:-1, :-1]
           
        dUxdx_GP1= dN1_dxy_GP1[0][0]*UxN1+ dN2_dxy_GP1[0][0]*UxN2+ dN3_dxy_GP1[0][0]*UxN3+ dN4_dxy_GP1[0][0]*UxN4
        dUxdy_GP1= dN1_dxy_GP1[1][0]*UxN1+ dN2_dxy_GP1[1][0]*UxN2+ dN3_dxy_GP1[1][0]*UxN3+ dN4_dxy_GP1[1][0]*UxN4
        dUydx_GP1= dN1_dxy_GP1[0][0]*UyN1+ dN2_dxy_GP1[0][0]*UyN2+ dN3_dxy_GP1[0][0]*UyN3+ dN4_dxy_GP1[0][0]*UyN4
        dUydy_GP1= dN1_dxy_GP1[1][0]*UyN1+ dN2_dxy_GP1[1][0]*UyN2+ dN3_dxy_GP1[1][0]*UyN3+ dN4_dxy_GP1[1][0]*UyN4
                   
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_xy_GP1= 0.5*(dUydx_GP1+dUxdy_GP1)
        e_xx_GP1_hat = e_xx_GP1.detach()
        e_yy_GP1_hat = e_yy_GP1.detach()
        e_xy_GP1_hat = e_xy_GP1.detach()
        S_xx_GP1_d= (E*(e_xx_GP1+ nu*e_yy_GP1)/(1-nu**2))        
        S_yy_GP1_d= (E*(e_yy_GP1+ nu*e_xx_GP1)/(1-nu**2))
        S_xy_GP1_d= (E*e_xy_GP1/(1+nu))
        S_xx_GP1_c= (E_hat*(e_xx_GP1_hat+ nu*e_yy_GP1_hat)/(1-nu**2))        
        S_yy_GP1_c= (E_hat*(e_yy_GP1_hat+ nu*e_xx_GP1_hat)/(1-nu**2))
        S_xy_GP1_c= (E_hat*e_xy_GP1_hat/(1+nu))
        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1_d+ e_yy_GP1*S_yy_GP1_d+ 2*e_xy_GP1*S_xy_GP1_d)
        compliance_GP1 = (e_xx_GP1_hat*S_xx_GP1_c+ e_yy_GP1_hat*S_yy_GP1_c+ 2*e_xy_GP1_hat*S_xy_GP1_c)

        ## GP2
        dUxdx_GP2= dN1_dxy_GP2[0][0]*UxN1+ dN2_dxy_GP2[0][0]*UxN2+ dN3_dxy_GP2[0][0]*UxN3+ dN4_dxy_GP2[0][0]*UxN4
        dUxdy_GP2= dN1_dxy_GP2[1][0]*UxN1+ dN2_dxy_GP2[1][0]*UxN2+ dN3_dxy_GP2[1][0]*UxN3+ dN4_dxy_GP2[1][0]*UxN4
        dUydx_GP2= dN1_dxy_GP2[0][0]*UyN1+ dN2_dxy_GP2[0][0]*UyN2+ dN3_dxy_GP2[0][0]*UyN3+ dN4_dxy_GP2[0][0]*UyN4
        dUydy_GP2= dN1_dxy_GP2[1][0]*UyN1+ dN2_dxy_GP2[1][0]*UyN2+ dN3_dxy_GP2[1][0]*UyN3+ dN4_dxy_GP2[1][0]*UyN4

        e_xx_GP2= dUxdx_GP2
        e_yy_GP2= dUydy_GP2
        e_xy_GP2= 0.5*(dUydx_GP2+dUxdy_GP2)
        e_xx_GP2_hat = e_xx_GP2.detach()
        e_yy_GP2_hat = e_yy_GP2.detach()
        e_xy_GP2_hat = e_xy_GP2.detach()
        S_xx_GP2_d= (E*(e_xx_GP2+ nu*e_yy_GP2)/(1-nu**2))        
        S_yy_GP2_d= (E*(e_yy_GP2+ nu*e_xx_GP2)/(1-nu**2))
        S_xy_GP2_d= (E*e_xy_GP2/(1+nu))
        S_xx_GP2_c= (E_hat*(e_xx_GP2_hat+ nu*e_yy_GP2_hat)/(1-nu**2))        
        S_yy_GP2_c= (E_hat*(e_yy_GP2_hat+ nu*e_xx_GP2_hat)/(1-nu**2))
        S_xy_GP2_c= (E_hat*e_xy_GP2_hat/(1+nu))
        strainEnergy_GP2= (e_xx_GP2*S_xx_GP2_d+ e_yy_GP2*S_yy_GP2_d+ 2*e_xy_GP2*S_xy_GP2_d)
        compliance_GP2 = (e_xx_GP2_hat*S_xx_GP2_c+ e_yy_GP2_hat*S_yy_GP2_c+ 2*e_xy_GP2_hat*S_xy_GP2_c)

        ## GP3
        dUxdx_GP3= dN1_dxy_GP3[0][0]*UxN1+ dN2_dxy_GP3[0][0]*UxN2+ dN3_dxy_GP3[0][0]*UxN3+ dN4_dxy_GP3[0][0]*UxN4
        dUxdy_GP3= dN1_dxy_GP3[1][0]*UxN1+ dN2_dxy_GP3[1][0]*UxN2+ dN3_dxy_GP3[1][0]*UxN3+ dN4_dxy_GP3[1][0]*UxN4
        dUydx_GP3= dN1_dxy_GP3[0][0]*UyN1+ dN2_dxy_GP3[0][0]*UyN2+ dN3_dxy_GP3[0][0]*UyN3+ dN4_dxy_GP3[0][0]*UyN4
        dUydy_GP3= dN1_dxy_GP3[1][0]*UyN1+ dN2_dxy_GP3[1][0]*UyN2+ dN3_dxy_GP3[1][0]*UyN3+ dN4_dxy_GP3[1][0]*UyN4
                
        e_xx_GP3= dUxdx_GP3
        e_yy_GP3= dUydy_GP3
        e_xy_GP3= 0.5*(dUydx_GP3+dUxdy_GP3)
        e_xx_GP3_hat = e_xx_GP3.detach()
        e_yy_GP3_hat = e_yy_GP3.detach()
        e_xy_GP3_hat = e_xy_GP3.detach()             
        S_xx_GP3_d= (E*(e_xx_GP3+ nu*e_yy_GP3)/(1-nu**2))     
        S_yy_GP3_d= (E*(e_yy_GP3+ nu*e_xx_GP3)/(1-nu**2))
        S_xy_GP3_d= (E*e_xy_GP3/(1+nu))
        S_xx_GP3_c= (E_hat*(e_xx_GP3_hat+ nu*e_yy_GP3_hat)/(1-nu**2))     
        S_yy_GP3_c= (E_hat*(e_yy_GP3_hat+ nu*e_xx_GP3_hat)/(1-nu**2))
        S_xy_GP3_c= (E_hat*e_xy_GP3_hat/(1+nu))
        strainEnergy_GP3= (e_xx_GP3*S_xx_GP3_d+ e_yy_GP3*S_yy_GP3_d+ 2*e_xy_GP3*S_xy_GP3_d)
        compliance_GP3 = (e_xx_GP3_hat*S_xx_GP3_c+ e_yy_GP3_hat*S_yy_GP3_c+ 2*e_xy_GP3_hat*S_xy_GP3_c)

        ## GP4
        dUxdx_GP4= dN1_dxy_GP4[0][0]*UxN1+ dN2_dxy_GP4[0][0]*UxN2+ dN3_dxy_GP4[0][0]*UxN3+ dN4_dxy_GP4[0][0]*UxN4
        dUxdy_GP4= dN1_dxy_GP4[1][0]*UxN1+ dN2_dxy_GP4[1][0]*UxN2+ dN3_dxy_GP4[1][0]*UxN3+ dN4_dxy_GP4[1][0]*UxN4
        dUydx_GP4= dN1_dxy_GP4[0][0]*UyN1+ dN2_dxy_GP4[0][0]*UyN2+ dN3_dxy_GP4[0][0]*UyN3+ dN4_dxy_GP4[0][0]*UyN4
        dUydy_GP4= dN1_dxy_GP4[1][0]*UyN1+ dN2_dxy_GP4[1][0]*UyN2+ dN3_dxy_GP4[1][0]*UyN3+ dN4_dxy_GP4[1][0]*UyN4

        e_xx_GP4= dUxdx_GP4
        e_yy_GP4= dUydy_GP4
        e_xy_GP4= 0.5*(dUydx_GP4+dUxdy_GP4)
        e_xx_GP4_hat = e_xx_GP4.detach()
        e_yy_GP4_hat = e_yy_GP4.detach()
        e_xy_GP4_hat = e_xy_GP4.detach()         
        S_xx_GP4_d= (E*(e_xx_GP4+ nu*e_yy_GP4)/(1-nu**2))        
        S_yy_GP4_d= (E*(e_yy_GP4+ nu*e_xx_GP4)/(1-nu**2))
        S_xy_GP4_d= (E*e_xy_GP4/(1+nu))
        S_xx_GP4_c= (E_hat*(e_xx_GP4_hat+ nu*e_yy_GP4_hat)/(1-nu**2))        
        S_yy_GP4_c= (E_hat*(e_yy_GP4_hat+ nu*e_xx_GP4_hat)/(1-nu**2))
        S_xy_GP4_c= (E_hat*e_xy_GP4_hat/(1+nu))
        strainEnergy_GP4= (e_xx_GP4*S_xx_GP4_d+ e_yy_GP4*S_yy_GP4_d+ 2*e_xy_GP4*S_xy_GP4_d)
        compliance_GP4 = (e_xx_GP4_hat*S_xx_GP4_c+ e_yy_GP4_hat*S_yy_GP4_c+ 2*e_xy_GP4_hat*S_xy_GP4_c)

        # Strain energy at element
        SE = (( strainEnergy_GP1 +strainEnergy_GP2 +strainEnergy_GP3 +strainEnergy_GP4 ) * detJ * 0.5).flip(dims=[0]).T.flatten()
        compliance = (( compliance_GP1 +compliance_GP2 +compliance_GP3 +compliance_GP4 ) * detJ).flip(dims=[0]).T.flatten()
        SE_masked = SE[mask_col_elem]
        compliance_masked = compliance[mask_col_elem]
        strain_energy = torch.sum( SE_masked )
        loss_compliance = torch.sum( compliance_masked )
        loss_pde1 = torch.tensor(0.0)
        loss_pde2 = torch.tensor(0.0)

    elif Diff_type == 'quad4_reduced':
        detJ = model_list[0].detJ
        dN = model_list[0].dN
        dN1_dxy_GP1 = dN[0]
        dN2_dxy_GP1 = dN[1]
        dN3_dxy_GP1 = dN[2]
        dN4_dxy_GP1 = dN[3]

        Ux= u.reshape(Nx, Ny).T.flip(dims=[0])
        Uy= v.reshape(Nx, Ny).T.flip(dims=[0])
        
        UxN1= Ux[1:, :-1]
        UxN2= Ux[1:, 1:]
        UxN3= Ux[:-1, 1:]
        UxN4= Ux[:-1, :-1]

        UyN1= Uy[1:, :-1]
        UyN2= Uy[1:, 1:]
        UyN3= Uy[:-1, 1:]
        UyN4= Uy[:-1, :-1]
           
        dUxdx_GP1= dN1_dxy_GP1[0][0]*UxN1+ dN2_dxy_GP1[0][0]*UxN2+ dN3_dxy_GP1[0][0]*UxN3+ dN4_dxy_GP1[0][0]*UxN4
        dUxdy_GP1= dN1_dxy_GP1[1][0]*UxN1+ dN2_dxy_GP1[1][0]*UxN2+ dN3_dxy_GP1[1][0]*UxN3+ dN4_dxy_GP1[1][0]*UxN4
        dUydx_GP1= dN1_dxy_GP1[0][0]*UyN1+ dN2_dxy_GP1[0][0]*UyN2+ dN3_dxy_GP1[0][0]*UyN3+ dN4_dxy_GP1[0][0]*UyN4
        dUydy_GP1= dN1_dxy_GP1[1][0]*UyN1+ dN2_dxy_GP1[1][0]*UyN2+ dN3_dxy_GP1[1][0]*UyN3+ dN4_dxy_GP1[1][0]*UyN4
                   
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_xy_GP1= 0.5*(dUydx_GP1+dUxdy_GP1)
        e_xx_GP1_hat = e_xx_GP1.detach()
        e_yy_GP1_hat = e_yy_GP1.detach()
        e_xy_GP1_hat = e_xy_GP1.detach()
        S_xx_GP1_d= (E*(e_xx_GP1+ nu*e_yy_GP1)/(1-nu**2))        
        S_yy_GP1_d= (E*(e_yy_GP1+ nu*e_xx_GP1)/(1-nu**2))
        S_xy_GP1_d= (E*e_xy_GP1/(1+nu))
        S_xx_GP1_c= (E_hat*(e_xx_GP1_hat+ nu*e_yy_GP1_hat)/(1-nu**2))        
        S_yy_GP1_c= (E_hat*(e_yy_GP1_hat+ nu*e_xx_GP1_hat)/(1-nu**2))
        S_xy_GP1_c= (E_hat*e_xy_GP1_hat/(1+nu))
        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1_d+ e_yy_GP1*S_yy_GP1_d+ 2*e_xy_GP1*S_xy_GP1_d)
        compliance_GP1 = (e_xx_GP1_hat*S_xx_GP1_c+ e_yy_GP1_hat*S_yy_GP1_c+ 2*e_xy_GP1_hat*S_xy_GP1_c)

        # Strain energy at element
        SE = (4 * (strainEnergy_GP1) * detJ * 0.5).flip(dims=[0]).T.flatten()
        compliance = (4*( compliance_GP1 ) * detJ).flip(dims=[0]).T.flatten()
        SE_masked = SE[mask_col_elem]
        strain_energy = torch.sum( SE_masked )
        compliance_masked = compliance[mask_col_elem]
        loss_compliance = torch.sum( compliance_masked )
        loss_pde1 = torch.tensor(0.0)
        loss_pde2 = torch.tensor(0.0)

    else: # use numerical differentiation
        pass

    return loss_pde1, loss_pde2, strain_energy, external_work, loss_compliance, loss_mConstraint, massfrac, loss_cConstraint, costfrac, grey_fraction

############################### Define Parameters ##############################################
# define material properties and
D = [0, 0.4, 0.6, 1.0]  # density
E = [1e-5, 0.5, 0.7, 1.0] # modulus
P = [0, 1.6, 1.2, 1.0] # cost
nu = 0.3 # Poisson's ratio
p0 = 3 # initial penalty
pf = 3 # final penalty
massfrac_f = 0.2 # final mass fraction
costfrac_f = 0.3 # final mass fraction
frac_decrease = 0.5 # fraction of epoch to decrease volume fraction from VF0 to VF_f
b = 8 # sharpness parameters
thres_static = 0.5 # static threshold values to binarize the density field
rho_min = 0.1 # lower limit to define grey element (rho_min,rho_max)
rho_max = 0.9 # upper limit to define grey element (rho_min,rho_max)

#define structured elements
pad = 0 # padding layer thickness

# coarse mesh
Nelx = 100 # number of elements along x to define training dataset
Nely = 100 # number of elements along y to define training dataset
Nelx_max = 200
Nely_max = 200
Nelx_min = 100
Nely_min = 100

num_CP = 50
xmin = 0.0
xmax = 100.0
ymin = 0.0
ymax = 100.0
hole_center = [-50.0,-50.0]
hole_radius = -1.0

# define model parameters and training parameters
random_state = [1,3,5,7,9,11,13,15,17,19]
Example = 'EX2D4'
Case = 'cost'
init_method = 'kaiming_uniform_'#xavier_uniform_  xavier_normal_  kaiming_uniform_
dynamic_weight = False
gradient_clip = False
Diff_type = 'quad4_reduced' 
omega = 0.5
learning_rate_disp = 1e-3
learning_rate_rho = 1e-4
nrmThreshold = 0.1; # maximum allowable norm of the gradients, [-nrmThreshold,nrmThreshold]
TO_num_iter = 10000 # number of total topology optimization iteration
plot_num = 10 # number of plots to save
delta = 1e-1
wc = 1 # weight factor for loss_compliance
wd = 1e3 # weight factor for loss_dem
wm = 0.0 # weight factor for mass constraint 
wp = 1e3 # weight factor for cost constraint    
basis = 'PGCAN'#'PGCAN'#'neural_network' #'M3'  
basis_rho = 'PGCAN'#'PGCAN'#'neural_network' #'M3'  
quant_correlation_class = 'Rough_RBF' 
activation = 'tanh'
if basis == 'PGCAN':
    n_features = 128 # must be a even number, only used for PGCAN
    n_cells = 3 # no more than 4, only used for PGCAN
    res = [36,36] # only for PGCAN
    n_neurons = int(n_features/2)
    n_layers = 3 # 3 layers at most for PGCAN
    NN_arch = [n_neurons] * n_layers 
    kernel_size = (2,2)
else:
    n_features = None #only used for PGCAN
    n_cells = None #only used for PGCAN
    res = None # only for PGCAN
    n_neurons = 64
    n_layers = 6
    NN_arch = [n_neurons] * n_layers 
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
domain = {'x':[xmin, xmax], 'y':[ymin, ymax]}
MP = {'hole_center':hole_center,'hole_radius':hole_radius,'pad':pad,'Nelx': Nelx, 'Nely': Nely,'Nelx_max': Nelx_max, 'Nely_max': Nely_max, 'Nelx_min': Nelx_min, 'Nely_min': Nely_min, 'num_CP':num_CP,
      'domain': domain,'thres_static':thres_static,'Diff_type':Diff_type,'num_phase':num_phase,
      'D':D_tensor,'E': E_tensor,'P':P_tensor, 'nu': nu,'p':p0,'p0':p0,'pf':pf,'b':b,'rho_min':rho_min,'rho_max':rho_max,
      'massfrac_star':massfrac_f, 'massfrac0':massfrac_f, 'massfrac_f':massfrac_f,'frac_decrease':frac_decrease,
      'costfrac_star':costfrac_f, 'costfrac0':costfrac_f, 'costfrac_f':costfrac_f,
      'frac_step_mass':frac_step,'frac_step_cost':frac_step,'frac_step_p':frac_step}
NN_config_disp = {'init_method':init_method,'random_state':random_state,'state':[],'Example':Example,'Case':Case,
             'dynamic_weight':dynamic_weight,'gradient_clip':gradient_clip,
             'omega':omega,'learning_rate_disp':learning_rate_disp,'learning_rate_rho':learning_rate_rho,'nrmThreshold':nrmThreshold,
             'TO_num_iter':TO_num_iter,'plot_num':plot_num,'plotting_interval_TO':plotting_interval_TO,
             'delta':delta,'wc':wc,'wd':wd,'wm':wm,'wp':wp,'kernel_size':kernel_size,
             'basis': basis, 'quant_correlation_class': quant_correlation_class, 'activation': activation,
             'n_features': n_features, 'n_cells': n_cells, 'res': res,'NN_arch': NN_arch, 'save_folder': []}
# construct NN_config for the density network
NN_config_rho = NN_config_disp
NN_config_rho['basis'] = basis_rho
if basis_rho == 'PGCAN':
    NN_config_rho['n_features'] = 128 # must be a even number, only used for PGCAN
    NN_config_rho['n_cells'] = 6 # no more than 4, only used for PGCAN
    NN_config_rho['res'] = [36,36] # only for PGCAN
    n_neurons = int(NN_config_rho['n_features']/2)
    n_layers = 3 # 3 layers at most for PGCAN
    NN_config_rho['NN_arch'] = [n_neurons] * n_layers 
    NN_config_rho['kernel_size'] = (2,2)
else:
    NN_config_rho['n_features'] = None #only used for PGCAN
    NN_config_rho['n_cells'] = None #only used for PGCAN
    NN_config_rho['res'] = None # only for PGCAN
    n_neurons = 64
    n_layers = 6
    NN_config_rho['NN_arch'] = [n_neurons] * n_layers 

############################### Generate Data ##############################################
Training, X_col_all = get_data(MP)
MP['domain_volume'] = Training['domain_volume']
X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

# Generate random integers
index_CP = np.random.randint(0, num_CP, size=TO_num_iter)

############################### TO Loop ##########################################
base_folder = f"/home/alexsunuci/PIGP2D_Aug14th_2025/Results/{Example}/{Case}/"
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
                num_output = 2).to(tkwargs['device'])

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                NN_config = NN_config_disp,
                Training = Training,
                name_output='v',
                MP=MP,
                num_output = 2).to(tkwargs['device'])
    
    model_phase = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                NN_config = NN_config_rho,
                Training = Training,
                name_output='rho',
                MP=MP,
                num_output = num_phase).to(tkwargs['device'])
    
    model_list = [model_u, model_v, model_phase]

    # define the time history dict
    timeHistory = {'loss_total':[],'loss_compliance':[],'loss_dem':[],'loss_mConstraint':[],'loss_cConstraint':[],'loss_tv':[],
                   'strain_energy':[],'external_work':[],'loss_pde_1':[],'loss_pde_2':[],'massfrac':[],'costfrac':[],
                   'wc':[], 'wd':[], 'wm':[],'wp':[],'grey':[],'p':[]}  
    
    # Define optimizer for just the two models
    optimizer_disp = torch.optim.Adam(model_list[0].parameters(),lr=learning_rate_disp,amsgrad=True)
    optimizer_rho = torch.optim.Adam(model_list[2].parameters(),lr=learning_rate_rho,amsgrad=True)
   
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
        # zero gradients from previous iteration
        optimizer_disp.zero_grad()
        optimizer_rho.zero_grad()

        # update parameters and collocation points
        index = index_CP[epoch]
        model_list[0].dx = X_col_all[index]['dx']
        model_list[0].dy = X_col_all[index]['dy']
        model_list[0].Nx = X_col_all[index]['Nx']
        model_list[0].Ny = X_col_all[index]['Ny']
        model_list[0].mask_col = X_col_all[index]['mask_col']
        model_list[0].mask_col_elem = X_col_all[index]['mask_col_elem']
        model_list[0].collocation_x = X_col_all[index]['X_col']
        model_list[0].elem_x = X_col_all[index]['X_col_elem']
        model_list[0].traction_indices = X_col_all[index]['traction_indices']
        model_list[0].traction_magnitude = X_col_all[index]['traction_magnitude']
        model_list[0].detJ = X_col_all[index]['detJ']
        model_list[0].dN = X_col_all[index]['dN']

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
        loss_pde1, loss_pde2, strain_energy, external_work, loss_compliance, loss_mConstraint, massfrac, loss_cConstraint, costfrac, grey_fraction = calculate_TO_loss(model_list,Diff_type)

        offset_dem = (1 + delta)/2 * external_work.detach()
        loss_dem = (strain_energy - external_work + offset_dem)
        
        if dynamic_weight:
            alpha = compute_dynamic_weight(loss_dem, loss_mConstraint, model_list)
            if torch.is_tensor(alpha):
                if not torch.isnan(alpha).any() and not torch.isinf(alpha).any():
                    model_list[2].alpha = alpha.item()

        loss =  wc * loss_compliance + wd * loss_dem + wm * loss_mConstraint + wp * loss_cConstraint
        loss.backward(retain_graph=True)
        # Clip gradients to avoid NaN or inf values
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
            torch.nn.utils.clip_grad_norm_(model_list[2].parameters(), nrmThreshold)

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
        timeHistory['loss_pde_1'].append(loss_pde1.item()) 
        timeHistory['loss_pde_2'].append(loss_pde2.item()) 
        timeHistory['wc'].append(wc) 
        timeHistory['wd'].append(wd) 
        timeHistory['wm'].append(wm) 
        timeHistory['wp'].append(wp) 
        timeHistory['grey'].append(grey_fraction) 
        timeHistory['p'].append(model_list[0].MP['p']) 
        
        # visualize contours and history
        if  (epoch+1) % plotting_interval_TO == 0:
            end_time = time.time() 
            # data = eval_model(model_list,Diff_type)
            # plot_all(epoch,save_folder,data,timeHistory)
            torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[2].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_{epoch}.pth')
            total_time = total_time + (end_time - start_time)
            start_time = time.time()

        model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
        model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
        model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])

    end_time = time.time()
    total_time = total_time + (end_time - start_time)

    # save the MP to a JSON file
    torch.save(MP, save_folder + "MP.pt")  # binary format
    
    # save the NN_config to a JSON file
    with open(save_folder + "NN_config_disp.json", "w") as file:
        json.dump(NN_config_disp, file, indent=4)
    with open(save_folder + "NN_config_rho.json", "w") as file:
        json.dump(NN_config_rho, file, indent=4)

    # save the Training to file
    torch.save(Training, save_folder + "Training.pth")

    # save the time history
    with open(save_folder + "timeHistory.json", "w") as file:
        json.dump(timeHistory, file)
    
    # save the final model
    # torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[3].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_final.pth')

    # pred(save_folder)

    # Write the loss terms to the file
    result_file_path = f"{save_folder}Results_summary.txt"
    with open(result_file_path, 'w') as f:
        f.write(f"The total training time in second is: {total_time}\n")
        f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
        f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")

