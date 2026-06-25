import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from models.lmgp import LMGP 
from utils.utils_general import get_tkwargs, projectDensity
import matplotlib.pyplot as plt
from gpytorch.settings import cholesky_jitter
from utils.utils_general import central_diff_2nd as ND2
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

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

def get_dN_2D(Jinv,Diff_type):
    if Diff_type == 'quad4':
        ## Differentiation of shape functions at gauss quadrature points       
        dN1_dsy=np.array([[-0.394337567297406, -0.394337567297406, -0.105662432702594, -0.105662432702594],
                              [-0.394337567297406, -0.105662432702594, -0.105662432702594, -0.394337567297406]])
        dN2_dsy=np.array([[0.394337567297406, 0.394337567297406, 0.105662432702594, 0.105662432702594],
                              [-0.105662432702594, -0.394337567297406, -0.394337567297406, -0.105662432702594]])
        dN3_dsy=np.array([[0.105662432702594, 0.105662432702594, 0.394337567297406, 0.394337567297406],
                              [0.105662432702594, 0.394337567297406, 0.394337567297406, 0.105662432702594]])
        dN4_dsy=np.array([[-0.105662432702594, -0.105662432702594, -0.394337567297406, -0.394337567297406],
                              [0.394337567297406, 0.105662432702594, 0.105662432702594, 0.394337567297406]])

        dN1_dxy_GP1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0]]).reshape((2,1)))
        dN2_dxy_GP1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0]]).reshape((2,1)))
        dN3_dxy_GP1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0]]).reshape((2,1)))
        dN4_dxy_GP1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0]]).reshape((2,1)))

        dN1_dxy_GP2=np.matmul(Jinv, np.array([dN1_dsy[0][1],dN1_dsy[1][1]]).reshape((2,1)))
        dN2_dxy_GP2=np.matmul(Jinv, np.array([dN2_dsy[0][1],dN2_dsy[1][1]]).reshape((2,1)))
        dN3_dxy_GP2=np.matmul(Jinv, np.array([dN3_dsy[0][1],dN3_dsy[1][1]]).reshape((2,1)))
        dN4_dxy_GP2=np.matmul(Jinv, np.array([dN4_dsy[0][1],dN4_dsy[1][1]]).reshape((2,1)))

        dN1_dxy_GP3=np.matmul(Jinv, np.array([dN1_dsy[0][2],dN1_dsy[1][2]]).reshape((2,1)))
        dN2_dxy_GP3=np.matmul(Jinv, np.array([dN2_dsy[0][2],dN2_dsy[1][2]]).reshape((2,1)))
        dN3_dxy_GP3=np.matmul(Jinv, np.array([dN3_dsy[0][2],dN3_dsy[1][2]]).reshape((2,1)))
        dN4_dxy_GP3=np.matmul(Jinv, np.array([dN4_dsy[0][2],dN4_dsy[1][2]]).reshape((2,1)))

        dN1_dxy_GP4=np.matmul(Jinv, np.array([dN1_dsy[0][3],dN1_dsy[1][3]]).reshape((2,1)))
        dN2_dxy_GP4=np.matmul(Jinv, np.array([dN2_dsy[0][3],dN2_dsy[1][3]]).reshape((2,1)))
        dN3_dxy_GP4=np.matmul(Jinv, np.array([dN3_dsy[0][3],dN3_dsy[1][3]]).reshape((2,1)))
        dN4_dxy_GP4=np.matmul(Jinv, np.array([dN4_dsy[0][3],dN4_dsy[1][3]]).reshape((2,1)))

        dN = [dN1_dxy_GP1, dN1_dxy_GP2, dN1_dxy_GP3, dN1_dxy_GP4,
              dN2_dxy_GP1, dN2_dxy_GP2, dN2_dxy_GP3, dN2_dxy_GP4,
              dN3_dxy_GP1, dN3_dxy_GP2, dN3_dxy_GP3, dN3_dxy_GP4,
              dN4_dxy_GP1, dN4_dxy_GP2, dN4_dxy_GP3, dN4_dxy_GP4]
    else: #'quad4_reduced':
        dN1_dsy=np.array([[-0.25],
                            [-0.25]])
        dN2_dsy=np.array([[0.25],
                            [-0.25]])
        dN3_dsy=np.array([[0.25],
                            [0.25]])
        dN4_dsy=np.array([[-0.25],
                            [0.25]])
        dN1_dxy_GP1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0]]).reshape((2,1)))
        dN2_dxy_GP1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0]]).reshape((2,1)))
        dN3_dxy_GP1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0]]).reshape((2,1)))
        dN4_dxy_GP1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0]]).reshape((2,1)))
        dN = [dN1_dxy_GP1, dN2_dxy_GP1, dN3_dxy_GP1, dN4_dxy_GP1]
    
    return dN

def eval_model(model_list,Diff_type):
    num_phase = model_list[0].MP['num_phase']
    D = model_list[0].MP['D']  # e.g., [0, 0.4, 0.6, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    elem_volume = dx * dy
    E_tensor = model_list[0].MP['E'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor

    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    massfrac_star = model_list[0].MP['massfrac_star']
    Nx = model_list[0].Nx
    Ny = model_list[0].Ny
    mask_col = model_list[0].mask_col
    collocation_x = model_list[0].collocation_x.clone()
    mask_col_elem = model_list[0].mask_col_elem
    elem_x = model_list[0].elem_x.clone()
    
    for model in model_list:
        model.eval

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
    rho = (phase_weights * D).sum(dim=1)  # shape: [n_elem]
    rho_elem_grid = rho.reshape(Nx-1, Ny-1).T.flip(dims=[0])
    # calculate the young's modulus
    E_tensor = model_list[0].MP['E'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor on the same device
    phase_weights_no_grad = phase_weights.detach() # shape: [n_elem, n_phases]
    E_hat_node = (E_tensor * (phase_weights_no_grad)**(2*p)/((phase_weights + 1e-8) ** p)).sum(dim = 1) # shape: [n_elem]
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
        S_xx_GP1_c= (E_hat*(e_xx_GP1_hat+ nu*e_yy_GP1_hat)/(1-nu**2))        
        S_yy_GP1_c= (E_hat*(e_yy_GP1_hat+ nu*e_xx_GP1_hat)/(1-nu**2))
        S_xy_GP1_c= (E_hat*e_xy_GP1_hat/(1+nu))
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
        S_xx_GP2_c= (E_hat*(e_xx_GP2_hat+ nu*e_yy_GP2_hat)/(1-nu**2))        
        S_yy_GP2_c= (E_hat*(e_yy_GP2_hat+ nu*e_xx_GP2_hat)/(1-nu**2))
        S_xy_GP2_c= (E_hat*e_xy_GP2_hat/(1+nu))
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
        S_xx_GP3_c= (E_hat*(e_xx_GP3_hat+ nu*e_yy_GP3_hat)/(1-nu**2))     
        S_yy_GP3_c= (E_hat*(e_yy_GP3_hat+ nu*e_xx_GP3_hat)/(1-nu**2))
        S_xy_GP3_c= (E_hat*e_xy_GP3_hat/(1+nu))
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
        S_xx_GP4_c= (E_hat*(e_xx_GP4_hat+ nu*e_yy_GP4_hat)/(1-nu**2))        
        S_yy_GP4_c= (E_hat*(e_yy_GP4_hat+ nu*e_xx_GP4_hat)/(1-nu**2))
        S_xy_GP4_c= (E_hat*e_xy_GP4_hat/(1+nu))
        compliance_GP4 = (e_xx_GP4_hat*S_xx_GP4_c+ e_yy_GP4_hat*S_yy_GP4_c+ 2*e_xy_GP4_hat*S_xy_GP4_c)

        # Strain energy at element
        compliance = (( compliance_GP1 +compliance_GP2 +compliance_GP3 +compliance_GP4 ) * detJ)
        # dC_drho = torch.autograd.grad(compliance, rho_elem_grid, torch.ones_like(compliance), create_graph=True)[0]
        compliance = compliance.flip(dims=[0]).T.flatten()
        # dC_drho = dC_drho.flip(dims=[0]).T.flatten()

        e11 = ((e_xx_GP1 + e_xx_GP2 + e_xx_GP3 + e_xx_GP4)/4).flip(dims=[0]).T.flatten()
        e22 = ((e_yy_GP1 + e_yy_GP2 + e_yy_GP3 + e_yy_GP4)/4).flip(dims=[0]).T.flatten()
        e12 = ((e_xy_GP1 + e_xy_GP2 + e_xy_GP3 + e_xy_GP4)/4).flip(dims=[0]).T.flatten()
        s11 = ((S_xx_GP1_c + S_xx_GP2_c + S_xx_GP3_c + S_xx_GP4_c)/4).flip(dims=[0]).T.flatten()
        s22 = ((S_yy_GP1_c + S_yy_GP2_c + S_yy_GP3_c + S_yy_GP4_c)/4).flip(dims=[0]).T.flatten()
        s12 = ((S_xy_GP1_c + S_xy_GP2_c + S_xy_GP3_c + S_xy_GP4_c)/4).flip(dims=[0]).T.flatten()

        u = u.squeeze(-1)
        v = v.squeeze(-1)
        u = u[mask_col].detach().cpu().numpy()
        v = v[mask_col].detach().cpu().numpy()
        elem_x = elem_x[mask_col_elem].detach().cpu().numpy()
        e11 = e11[mask_col_elem].detach().cpu().numpy()
        e22 = e22[mask_col_elem].detach().cpu().numpy()
        e12 = e12[mask_col_elem].detach().cpu().numpy()
        s11 = s11[mask_col_elem].detach().cpu().numpy()
        s22 = s22[mask_col_elem].detach().cpu().numpy()
        s12 = s12[mask_col_elem].detach().cpu().numpy()
        # rho = rho_elem_grid.flip(dims=[0]).T.flatten()
        rho = rho[mask_col_elem].detach().cpu().numpy()
        collocation_x = collocation_x[mask_col].detach().cpu().numpy()
        comp_vector = compliance[mask_col_elem].detach().cpu().numpy()
        # dC_drho = dC_drho[mask_col_elem].detach().cpu().numpy()
        residual_pde1 = None
        residual_pde2 = None

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
        S_xx_GP1_c= (E_hat*(e_xx_GP1_hat+ nu*e_yy_GP1_hat)/(1-nu**2))        
        S_yy_GP1_c= (E_hat*(e_yy_GP1_hat+ nu*e_xx_GP1_hat)/(1-nu**2))
        S_xy_GP1_c= (E_hat*e_xy_GP1_hat/(1+nu))
        compliance_GP1 = (e_xx_GP1_hat*S_xx_GP1_c+ e_yy_GP1_hat*S_yy_GP1_c+ 2*e_xy_GP1_hat*S_xy_GP1_c)

        # Strain energy at element
        compliance = (4*( compliance_GP1 ) * detJ)
        # dC_drho = None# torch.autograd.grad(compliance, rho_elem_grid, torch.ones_like(compliance), create_graph=True)[0]
        compliance = compliance.flip(dims=[0]).T.flatten()
        # dC_drho = None #dC_drho.flip(dims=[0]).T.flatten()
        e11 = ((e_xx_GP1)).flip(dims=[0]).T.flatten()
        e22 = ((e_yy_GP1)).flip(dims=[0]).T.flatten()
        e12 = ((e_xy_GP1)).flip(dims=[0]).T.flatten()
        s11 = ((S_xx_GP1_c)).flip(dims=[0]).T.flatten()
        s22 = ((S_yy_GP1_c)).flip(dims=[0]).T.flatten()
        s12 = ((S_xy_GP1_c)).flip(dims=[0]).T.flatten()

        u = u.squeeze(-1)
        v = v.squeeze(-1)
        u = u[mask_col].detach().cpu().numpy()
        v = v[mask_col].detach().cpu().numpy()
        # elem_x = elem_x[mask_col_elem].detach().cpu().numpy()
        elem_x = elem_x.detach().cpu().numpy()
        e11 = e11[mask_col_elem].detach().cpu().numpy()
        e22 = e22[mask_col_elem].detach().cpu().numpy()
        e12 = e12[mask_col_elem].detach().cpu().numpy()
        s11 = s11[mask_col_elem].detach().cpu().numpy()
        s22 = s22[mask_col_elem].detach().cpu().numpy()
        s12 = s12[mask_col_elem].detach().cpu().numpy()
        # rho = rho_elem_grid.flip(dims=[0]).T.flatten()
        # rho = rho[mask_col_elem].detach().cpu().numpy()
        rho = rho.detach().cpu().numpy()
        collocation_x = collocation_x[mask_col].detach().cpu().numpy()
        comp_vector = compliance[mask_col_elem].detach().cpu().numpy()
        # dC_drho = dC_drho[mask_col_elem].detach().cpu().numpy()
        residual_pde1 = None
        residual_pde2 = None

    else: # use numerical differentiation
        pass

    Data = {'mask_nan':model_list[0].mask_nan,'xi':model_list[0].xi,'yi':model_list[0].yi,
            'collocation_x':collocation_x,'u':u,'v':v,'rho':rho,'elem_x':elem_x,
            'residual_pde1':residual_pde1,'residual_pde2':residual_pde2, 'mask_col_elem':mask_col_elem,
            's11':s11,'s22':s22,'s12':s12,'e11':e11,'e22':e22,'e12':e12, 'comp_vector':comp_vector,}
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

def extract_density(save_folder, elem_num, iter):
    Nelx = elem_num[0]
    Nely = elem_num[1]
    # load the MP
    MP = torch.load(save_folder + 'MP.pt', map_location=torch.device('cpu'))

    # load the NN_config
    with open(save_folder + "NN_config_disp.json", "r") as file:
        NN_config_disp = json.load(file)
    with open(save_folder + "NN_config_rho.json", "r") as file:
        NN_config_rho = json.load(file)
    
    # load the training data:
    Training = torch.load(save_folder + "Training.pth",map_location=torch.device('cpu'))

    X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True)
    u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False)

    model_u = LMGP(train_x = u_X_train, train_y = u_train, collocation_x = X_col,
                NN_config = NN_config_disp,
                Training = Training,
                name_output='u',
                MP=MP,
                num_output = 2)

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                NN_config = NN_config_disp,
                Training = Training,
                name_output='v',
                MP=MP,
                num_output = 2)

    model_rho = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                NN_config = NN_config_rho,
                Training = Training,
                name_output='rho',
                MP=MP,
                num_output = MP['num_phase'])

    model_list = [model_u, model_v, model_rho]
    checkpoint = torch.load(save_folder + f'Trained_models_{iter}.pth',map_location=torch.device('cpu'))
    model_list[0].mean_module_NN_All.load_state_dict(checkpoint['model_u_state_dict'])
    model_list[2].mean_module_NN_All.load_state_dict(checkpoint['model_rho_state_dict'])

    domain = MP['domain']
    pad = MP['pad']
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xmin, xmax,ymin,ymax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi = np.pad(xi, pad_width=pad, mode='linear_ramp', end_values=(xi[0] - pad*dx, xi[-1] + pad*dx))
    yi = np.pad(yi, pad_width=pad, mode='linear_ramp', end_values=(yi[0] - pad*dy, yi[-1] + pad*dy))
    xi, yi = np.meshgrid(xi, yi)

    # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2) ** 0.5
    mask_domain = distance >= -1
    mask_col = mask_domain
    mask_nan = ~mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    #mask_nan = mask_nan.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    # create meshgrid for fem element center
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1]
    distance = ((xi_elem - hole_center[0]) ** 2 + (yi_elem- hole_center[1]) ** 2) ** 0.5
    mask_hole_elem = (distance < hole_radius)
    mask_domain_elem = (distance >= hole_radius) # mask domain contains hole points, use GP on those points 
    mask_col_elem = mask_domain_elem.T.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.T.flatten(),yi_elem.T.flatten()]).T)
    X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    # define constants for element mesh
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # Save all data in a dictionary
    X_col_data = {'X_col': X_col,'mask_col': mask_col,'mask_nan': mask_nan,'xi':xi,'yi':yi,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'detJ':detJ,'dN':dN}
    model_list[0].dx = X_col_data['dx']
    model_list[0].dy = X_col_data['dy']
    model_list[0].Nx = X_col_data['Nx']
    model_list[0].Ny = X_col_data['Ny']
    model_list[0].xi = X_col_data['xi']
    model_list[0].yi = X_col_data['yi']
    model_list[0].mask_col = X_col_data['mask_col']
    model_list[0].collocation_x = X_col_data['X_col']
    model_list[0].mask_nan = X_col_data['mask_nan']
    model_list[0].mask_col_elem = X_col_data['mask_col_elem']
    model_list[0].elem_x = X_col_data['X_col_elem']
    model_list[0].detJ = X_col_data['detJ']
    model_list[0].dN = X_col_data['dN']
    data = eval_model(model_list,MP['Diff_type'])
    return data

def extract_density_hole(save_folder, elem_num, iter):
    Nelx = elem_num[0]
    Nely = elem_num[1]
    # load the MP
    MP = torch.load(save_folder + 'MP.pt', map_location=torch.device('cpu'))

    # load the NN_config
    with open(save_folder + "NN_config_disp.json", "r") as file:
        NN_config_disp = json.load(file)
    with open(save_folder + "NN_config_rho.json", "r") as file:
        NN_config_rho = json.load(file)
    
    # load the training data:
    Training = torch.load(save_folder + "Training.pth",map_location=torch.device('cpu'))

    X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True)
    u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False)

    model_u = LMGP(train_x = u_X_train, train_y = u_train, collocation_x = X_col,
                NN_config = NN_config_disp,
                Training = Training,
                name_output='u',
                MP=MP,
                num_output = 2)

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                NN_config = NN_config_disp,
                Training = Training,
                name_output='v',
                MP=MP,
                num_output = 2)

    model_rho = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                NN_config = NN_config_rho,
                Training = Training,
                name_output='rho',
                MP=MP,
                num_output = MP['num_phase'])

    model_list = [model_u, model_v, model_rho]
    checkpoint = torch.load(save_folder + f'Trained_models_{iter}.pth',map_location=torch.device('cpu'))
    model_list[0].mean_module_NN_All.load_state_dict(checkpoint['model_u_state_dict'])
    model_list[2].mean_module_NN_All.load_state_dict(checkpoint['model_rho_state_dict'])

    domain = MP['domain']
    pad = MP['pad']
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xmin, xmax,ymin,ymax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi = np.pad(xi, pad_width=pad, mode='linear_ramp', end_values=(xi[0] - pad*dx, xi[-1] + pad*dx))
    yi = np.pad(yi, pad_width=pad, mode='linear_ramp', end_values=(yi[0] - pad*dy, yi[-1] + pad*dy))
    xi, yi = np.meshgrid(xi, yi)

    # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2) ** 0.5
    mask_domain = distance >= -1
    mask_col = mask_domain
    mask_nan = ~mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    #mask_nan = mask_nan.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    # create meshgrid for fem element center
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1]
    distance = ((xi_elem - hole_center[0]) ** 2 + (yi_elem- hole_center[1]) ** 2) ** 0.5
    mask_hole_elem = (distance < hole_radius)
    # mask_domain_elem = (distance >= hole_radius) # mask domain contains hole points, use GP on those points 
    mask_domain_elem = (xi_elem <= 40.0) | (yi_elem <= 40.0)
    mask_col_elem = mask_domain_elem.T.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.T.flatten(),yi_elem.T.flatten()]).T)
    X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    # define constants for element mesh
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # Save all data in a dictionary
    X_col_data = {'X_col': X_col,'mask_col': mask_col,'mask_nan': mask_nan,'xi':xi,'yi':yi,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'detJ':detJ,'dN':dN}
    model_list[0].dx = X_col_data['dx']
    model_list[0].dy = X_col_data['dy']
    model_list[0].Nx = X_col_data['Nx']
    model_list[0].Ny = X_col_data['Ny']
    model_list[0].xi = X_col_data['xi']
    model_list[0].yi = X_col_data['yi']
    model_list[0].mask_col = X_col_data['mask_col']
    model_list[0].collocation_x = X_col_data['X_col']
    model_list[0].mask_nan = X_col_data['mask_nan']
    model_list[0].mask_col_elem = X_col_data['mask_col_elem']
    model_list[0].elem_x = X_col_data['X_col_elem']
    model_list[0].detJ = X_col_data['detJ']
    model_list[0].dN = X_col_data['dN']
    data = eval_model(model_list,MP['Diff_type'])
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

tkwargs = get_tkwargs()

# EX2D2
Nelx = 800
Nely = 400
elem_num = [Nelx,Nely]

save_folder = 'Results/EX2D2/mass/'
iter = [999,1999,2999,3999,4999,5999,6999,7999,8999,9999]
iter = [9999]

for run_num in range(1,6):
    run_folder = os.path.join(save_folder, f'Run_{run_num}/')

    data_list = []
    for _,  i in enumerate(iter):
        # Extract data for each iteration
        data = extract_density(run_folder, elem_num, i)
        mask_col_elem = data['mask_col_elem']
        mask_nan = ~mask_col_elem
        rho = data['rho']
        rho[rho>1] = 1.0
        rho[rho<0] = 0.0
        rho[mask_nan] = 0.0
        elem_x = data['elem_x']
        x_elem = elem_x[:, 0]
        y_elem = elem_x[:, 1]
        xi = data['xi']
        yi = data['yi']
        zi_rho = griddata((x_elem, y_elem), rho, (xi, yi), method='nearest')
        data_list.append({
            'rho': zi_rho,
            'xi': xi,
            'yi': yi,
            'rho_elem':rho,
            'elem_x': elem_x
        })

    # plot the distribution of rho
    # rho_flat = rho.flatten()  # or rho.ravel()
    # plt.figure(figsize=(6, 4))
    # plt.hist(rho_flat, bins=400, color='steelblue', edgecolor='black', density=True)
    # plt.xlabel('Density')
    # plt.ylabel('Probability Density')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # plot the two densities
    for i in range(len(data_list)):
        fig, axs = plt.subplots(figsize=(12, 6))
        anno_loc = [0.0, 0.9]
        num_levels = 500

        contour = axs.contourf(xi, yi, data_list[i]['rho'], levels=num_levels, cmap=cmap_phases, vmin=0, vmax=1)
        axs.set_rasterized(True)
        axs.set_title(f'Epoch: {iter[i]}')
        axs.set_aspect('equal')
        axs.set_xticks([])  # Hide x ticks
        axs.set_yticks([])  # Hide y ticks
        axs.set_xlabel('')  # Hide x label
        axs.set_ylabel('')  # Hide y label
        axs.axis('off')     # Remove bounding box

        cbar_axes = fig.add_axes([0.005, 0.15, 0.03, 0.65])  # Position [left, bottom, width, height]
        cbar = fig.colorbar(contour, cax=cbar_axes)  # Use cax for colorbar
        cbar.set_ticks(np.arange(0.0, 1.000001, 0.2))  # Set ticks on the colorbar
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        file_name = f"TO_evolution_density_{Nelx}_{Nely}_Epoch{iter[i]}.pdf"
        file_path = f"{run_folder}/{file_name}"
        plt.savefig(file_path, format='pdf', dpi=600)
        # plt.show()




