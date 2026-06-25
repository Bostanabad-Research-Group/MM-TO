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
from utils.get_training_data_3D import get_data_EX3D1 as get_data

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
    m_phase = model_list[1].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
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
    m_phase = model_list[1].mean_module_NN_All(elem_x)  # shape: [n_elem, n_phases]
    
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
    kappa_tensor = model_list[0].MP['kappa'] # e.g., [1e-9, 0.5, 0.7, 1.0] a tensor
    dx = model_list[0].dx
    dy = model_list[0].dy
    dz = model_list[0].dz
    elem_volume = dx * dy * dz

    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    s = model_list[0].MP['s']
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
    m_phase = model_list[1].mean_module_NN_All(elem_x)

    # g_u = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()
    # g_v = (model_list[1].covar_module(model_list[1].train_inputs[0], collocation_x)).evaluate()
    # g_w = (model_list[2].covar_module(model_list[2].train_inputs[0], collocation_x)).evaluate()
    # g_rho = (model_list[3].covar_module(model_list[3].train_inputs[0], elem_x)).evaluate()
    g_T = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()
            # model_list[1].chol_decomp = model_list[1].covar_module(model_list[1].train_inputs[0]).cholesky()
            # model_list[2].chol_decomp = model_list[2].covar_module(model_list[2].train_inputs[0]).cholesky()
            # model_list[3].chol_decomp = model_list[3].covar_module(model_list[3].train_inputs[0]).cholesky()

    K_inv_offset_T = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))
    
    # K_inv_offset_rho = model_list[3].chol_decomp._cholesky_solve(model_list[3].train_targets.unsqueeze(-1) - torch.sigmoid(model_list[3].mean_module_NN_All(model_list[3].train_inputs[0])[:,0].unsqueeze(-1)))

    T = (m_col[:,0].unsqueeze(-1) + g_T.t() @ K_inv_offset_T).squeeze(-1)
    # v = (m_col[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v).squeeze(-1)
    # w = (m_col[:,2].unsqueeze(-1) + g_w.t() @ K_inv_offset_w).squeeze(-1)
    
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

    if Diff_type == 'AD': #autograd
        pass
    elif Diff_type == 'hex': 
        pass

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

        Node_T= T.reshape(Nx, Ny, Nz)

        UxN1= Node_T[:-1,:-1,:-1]
        UxN2= Node_T[1:,:-1,:-1]
        UxN3= Node_T[1:,1:,:-1]
        UxN4= Node_T[:-1,1:,:-1]
        UxN5= Node_T[:-1,:-1,1:]
        UxN6= Node_T[1:,:-1,1:]
        UxN7= Node_T[1:,1:,1:]
        UxN8= Node_T[:-1,1:,1:]

        T_elem = (UxN1 + UxN2 + UxN3 + UxN4 + UxN5 + UxN6 + UxN7 + UxN8)/8. # only valide for hex element with reduced integration

        dTdx_GP1= dN1_GP1[0][0]*UxN1 + dN2_GP1[0][0]*UxN2 + dN3_GP1[0][0]*UxN3 + dN4_GP1[0][0]*UxN4 + dN5_GP1[0][0]*UxN5 + dN6_GP1[0][0]*UxN6 + dN7_GP1[0][0]*UxN7 + dN8_GP1[0][0]*UxN8
        dTdy_GP1= dN1_GP1[1][0]*UxN1 + dN2_GP1[1][0]*UxN2 + dN3_GP1[1][0]*UxN3 + dN4_GP1[1][0]*UxN4 + dN5_GP1[1][0]*UxN5 + dN6_GP1[1][0]*UxN6 + dN7_GP1[1][0]*UxN7 + dN8_GP1[1][0]*UxN8
        dTdz_GP1= dN1_GP1[2][0]*UxN1 + dN2_GP1[2][0]*UxN2 + dN3_GP1[2][0]*UxN3 + dN4_GP1[2][0]*UxN4 + dN5_GP1[2][0]*UxN5 + dN6_GP1[2][0]*UxN6 + dN7_GP1[2][0]*UxN7 + dN8_GP1[2][0]*UxN8

        # calculat external work:
        sT_elem = s * T_elem
        sT = (8.0 * ( sT_elem ) * detJ).flatten()
        external_work = sT.sum()
        
        # calculate conductivity
        phase_weights_no_grad = phase_weights.detach() # shape: [n_elem, n_phases]
        kappa = (kappa_tensor * phase_weights_no_grad ** p).sum(dim = 1).unsqueeze(1) # shape: [n_elem,1]
        kappa_hat = (kappa_tensor * (phase_weights_no_grad)**(2*p)/((phase_weights + 1e-8) ** p)).sum(dim = 1).unsqueeze(1) # shape: [n_elem,1]
        kappa_elem = kappa.reshape((Nx-1),(Ny-1),(Nz-1)) 
        kappa_hat_elem = kappa_hat.reshape((Nx-1),(Ny-1),(Nz-1))

        # calculate strain energy
        dT_square = dTdx_GP1 ** 2 + dTdy_GP1 ** 2 + dTdz_GP1 ** 2
        dT_square_no_grad = dT_square.detach()
        SE = (8.0 * ( kappa_elem * dT_square ) * detJ * 0.5).flatten()
        compliance = (8.0 * ( kappa_hat_elem * dT_square_no_grad ) * detJ * 0.5).flatten()
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
kappa = [1e-5, 1.0]  # conductivity
P = [0, 1.0] # cost
source = 1e-5 # heat sink
source_f = 1e-5 # final value of heat sink
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
Nelz = 30 # number of elements along y to define training dataset

Nelx_max = 80
Nely_max = 40
Nelz_max = 40
Nelx_min = 40
Nely_min = 20
Nelz_min = 20
num_CP = 51
xmin = 0.0
xmax = 60.0
ymin = 0.0
ymax = 30.0
zmin = 0.0
zmax = 30.0
hole_center = None
hole_radius = None
ring_thickness = None

# define model parameters and training parameters
random_state = [1,3,5,7,9,11,13,15,17,19]
Example = 'EX3D1'
Case = 'single_material'
dynamic_weight = False
gradient_clip = False
Diff_type = 'hex_reduced' 
omega = 0.1
learning_rate_disp = 1e-3
learning_rate_rho = 1e-4
nrmThreshold = 0.1; # maximum allowable norm of the gradients, [-nrmThreshold,nrmThreshold]
TO_num_iter = 10000 # number of total topology optimization iteration
plot_num = 20 # number of plots to save
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
    res = [22,11,11] # only for PGCAN3D [x,y,z]
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
kappa_tensor = torch.tensor(kappa, dtype=tkwargs["dtype"], device=tkwargs['device'])  # shape: [n_phases]
frac_decrease = int(frac_decrease * TO_num_iter)
plotting_interval_TO = TO_num_iter/plot_num
frac_step = None
frac_step_source = (source_f - source) / frac_decrease
plotting_interval_TO = TO_num_iter/plot_num
domain = {'x':[xmin, xmax], 'y':[ymin, ymax], 'z':[zmin, zmax]}
MP = {'hole_center':hole_center,'hole_radius':hole_radius,'ring_thickness':ring_thickness,'pad':pad,
      'Nelx': Nelx, 'Nely': Nely,'Nelz': Nelz,'Nelx_max': Nelx_max, 'Nely_max': Nely_max, 'Nelz_max': Nelz_max, 'kappa': kappa_tensor,
      'Nelx_min': Nelx_min, 'Nely_min': Nely_min, 'Nelz_min': Nelz_min, 'num_CP':num_CP,'num_phase':num_phase,
      'domain': domain,'thres_static':thres_static,'Diff_type':Diff_type,
      'D':D_tensor,'E': E_tensor,'P':P_tensor, 'nu': nu,'p':p0,'pf':pf,'p0':p0,'rho_min':rho_min,'rho_max':rho_max,'s':source,'s_f':source_f,
      'massfrac_star':massfrac_f, 'massfrac0':massfrac_f, 'massfrac_f':massfrac_f,'frac_decrease':frac_decrease,
      'costfrac_star':costfrac_f, 'costfrac0':costfrac_f, 'costfrac_f':costfrac_f,
      'frac_step_mass':frac_step,'frac_step_cost':frac_step,'frac_step_p':frac_step,'frac_step_source': frac_step_source}

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
T_X_train = Training['T_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
T_train = Training['T_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

# Generate random integers
index_CP = np.random.randint(0, num_CP, size=TO_num_iter)

############################### TO Loop ##########################################
base_folder = f"/home/alexsunuci/PIGP3D_Dec24th_2025/Results/{Example}/{Case}/"
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
    model_T = LMGP(train_x = T_X_train, train_y = T_train, collocation_x = X_col,
                    NN_config = NN_config_disp,
                    Training = Training,
                    name_output='T',
                    MP=MP,
                    num_output = 1).to(tkwargs['device'])

    model_phase = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                    NN_config = NN_config_rho,
                    Training = Training,
                    name_output='rho',
                    MP=MP,
                    num_output = num_phase).to(tkwargs['device'])

    model_list = [model_T, model_phase]

    # define the time history dict
    timeHistory = {'loss_total':[],'loss_compliance':[],'loss_dem':[],'loss_mConstraint':[],'loss_cConstraint':[],'grey':[],
               'strain_energy':[],'external_work':[],'massfrac':[],'costfrac':[],'wc':[], 'wd':[], 'wm':[],'wp':[],'p':[],'s':[]}  
    
    # Define optimizer for just the two models
    optimizer_disp = torch.optim.Adam(model_list[0].parameters(),lr=learning_rate_disp,amsgrad=True)
    optimizer_rho = torch.optim.Adam(model_list[1].parameters(),lr=learning_rate_rho,amsgrad=True)
    
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
        # model_list[0].traction_indices = X_col_all[index]['traction_indices']
        # model_list[0].traction_magnitude = X_col_all[index]['traction_magnitude']
        
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
            torch.nn.utils.clip_grad_norm_(model_list[1].parameters(), nrmThreshold)

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
        timeHistory['s'].append(model_list[0].MP['s'])
        
        # visualize contours and history
        if  (epoch+1) % plotting_interval_TO == 0:
            end_time = time.time()
            # save the model_list[0] to dict
            gpu_id = 0  # Assuming you are using GPU 0
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            # torch.save(model_list[0].mean_module_NN_All.state_dict(),save_folder + f'Trained_mean_module_NN_params_{epoch}.pth')
            torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[1].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_{epoch}.pth')
            total_time = total_time + (end_time - start_time)
            start_time = time.time()

        model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
        model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
        model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])
        model_list[0].MP['s'] = min(model_list[0].MP['s_f'], model_list[0].MP['s'] + model_list[0].MP['frac_step_source'])

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
    torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[1].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_final.pth')

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


