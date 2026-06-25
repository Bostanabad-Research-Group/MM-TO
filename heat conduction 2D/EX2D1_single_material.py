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
from models.lmgp_updated import LMGP 
from gpytorch.settings import cholesky_jitter
from tqdm import tqdm
from utils.utils_general import set_seed0, get_multiGPU
from utils.utils_general import compute_dynamic_weight_2 as compute_dynamic_weight
from utils.get_training_data_2D_updated import get_data_EX2D1 as get_data

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def calculate_TO_loss(model_list):
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
        model.train()

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

    # calculate elemental temperature
    T_elem = Node_T[conn].reshape(N_elem_w, -1) # [N_elem_w, 4] for CSP4
    T_elem = T_elem.unsqueeze(1).unsqueeze(-1)               # [N_elem_w, 1, 4, 1]
    T_elem = T_elem.expand(-1, N.shape[1], -1, -1)     # [N_elem_w, N_int, 4, 1]
    
    # calculate external work
    sT = s * torch.matmul(N, T_elem).squeeze(-1)              # [N_elem_w, N_int, 1]
    sT_int = sT.squeeze(-1) * detJ * wt.unsqueeze(0)   # [N_elem_w, N_int]
    external_work = sT_int.sum()
    
    # calculate conductivity
    phase_weights_no_grad = phase_weights.detach() # shape: [n_elem, n_phases]
    kappa = (kappa_tensor * phase_weights_no_grad ** p).sum(dim = 1).unsqueeze(1) # shape: [n_elem,1]
    kappa_hat = (kappa_tensor * (phase_weights_no_grad)**(2*p)/((phase_weights + 1e-8) ** p)).sum(dim = 1).unsqueeze(1) # shape: [n_elem,1]
    
    # calculate strain energy
    dT = torch.matmul(B, T_elem).squeeze(-1) # [N_elem_w, N_int, 2]
    dT_square = (dT ** 2).sum(dim=-1)   # [N_elem_w, N_int]
    kdT2 =  kappa * dT_square  # [N_elem_w, N_int]
    kdT2_int = kdT2 * detJ * wt.unsqueeze(0) # [N_elem_w, N_int]
    strain_energy = 0.5 * kdT2_int.sum()

    # calculate compliance
    dT_hat_square = dT_square.detach() # [N_elem_w, N_int]
    kdT2_hat =  kappa_hat * dT_hat_square  # [N_elem_w, N_int]
    kdT2_hat_int = kdT2_hat * detJ * wt.unsqueeze(0) # [N_elem_w, N_int]
    compliance = kdT2_hat_int.sum()
    
    return strain_energy, external_work, compliance, mass, M0, cost, cost0, grey_counts, count_total 

# ========================
# Worker: single GPU debug
# ========================
def run_worker(MP, NN_config_disp, NN_config_rho,
               T_X_train, T_train,
               rho_X_train, rho_train,
               index_CP, mesh_data):

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
    gradient_clip = NN_config_disp["gradient_clip"]
    nrmThreshold = NN_config_disp["nrmThreshold"]
    dynamic_weight = NN_config_disp["dynamic_weight"]
    random_state = NN_config_disp["random_state"]
    learning_rate_disp = NN_config_disp["learning_rate_disp"]
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
        model_T = LMGP(T_X_train, T_train, NN_config_disp,
                       name_output="T", MP=MP, num_output=1).to(device)
        model_phase = LMGP(rho_X_train, rho_train, NN_config_rho,
                           name_output="rho", MP=MP, num_output=num_phase).to(device)

        # Keep all in list for BC handling
        model_list = [model_T, model_phase]

        # define the time history dict
        timeHistory = {
            'loss_total': [], 'loss_compliance': [], 'loss_dem': [],
            'loss_mConstraint': [], 'loss_cConstraint': [], 'loss_tv': [],
            'strain_energy': [], 'external_work': [],
            'loss_pde_1': [], 'loss_pde_2': [],
            'massfrac': [], 'costfrac': [],
            'wc': [], 'wd': [], 'wm': [], 'wp': [],
            'grey': [], 'p': [], 's':[]
        }

        # ========================
        # Optimizers + schedulers
        # ========================
        optimizer_T = torch.optim.Adam(model_T.parameters(),
                                          lr=learning_rate_disp, amsgrad=True)
        optimizer_rho = torch.optim.Adam(model_phase.parameters(),
                                         lr=learning_rate_rho, amsgrad=True)

        scheduler_T = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_T,
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
            optimizer_T.zero_grad()
            optimizer_rho.zero_grad()

            # --- mesh slice ---
            index = index_CP[epoch]
            mesh_key = f"Mesh_{index:02d}"
            GPU_key = "GPU0"  # single GPU

            # assign mesh data to model_u
            model_list[0].collocation_x = mesh_data[GPU_key][mesh_key]["X_node"]
            model_list[0].elem_x        = mesh_data[GPU_key][mesh_key]["X_elem"]
            model_list[0].elem_vol        = mesh_data[GPU_key][mesh_key]["elem_vol"]
            model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
            model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
            model_list[0].N             = mesh_data[GPU_key][mesh_key]["N"]
            model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]

            # --- calculate local losses ---
            strain_energy, external_work, compliance, mass, M0, cost, cost0, grey_counts, count_total = calculate_TO_loss(model_list)
            
            # for multi-GPU, assemble the global quantities here:
            strain_energy_global = strain_energy
            external_work_global = external_work
            compliance_global    = compliance
            mass_global          = mass
            M0_global            = M0
            cost_global          = cost
            cost0_global         = cost0
            grey_counts_global   = grey_counts
            count_total_global   = count_total

            # calculate fraction 
            massfrac = mass_global / M0_global
            costfrac = cost_global / cost0_global
            grey_fraction = grey_counts_global / count_total_global
            
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
            loss_compliance = compliance_global
            offset_dem = (1 + delta) / 2 * external_work_global.detach()
            loss_dem = strain_energy_global - external_work_global + offset_dem
            loss_mConstraint = torch.square(torch.max((massfrac / massfrac_star) - 1.0, 0.0*mass_global))
            loss_cConstraint = torch.square(torch.max((costfrac / costfrac_star) - 1.0, 0.0*cost_global))
            loss =  wc * loss_compliance + wd * loss_dem + wm * loss_mConstraint #+ wp * loss_cConstraint
            
            # --- backward ---
            loss.backward(retain_graph=True)

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model_T.parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_phase.parameters(), nrmThreshold)

            optimizer_T.step()
            optimizer_rho.step()
            scheduler_T.step()
            scheduler_rho.step()

            # save the loss histories
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
            timeHistory['grey'].append(grey_fraction.detach().cpu().tolist())
            timeHistory['p'].append(model_list[0].MP['p'])
            timeHistory['p'].append(model_list[0].MP['s']) 

            # --- checkpoint (rank 0 only) ---
            if  (epoch+1) % plotting_interval_TO == 0:
                end_time = time.time() 
                torch.save({'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),'model_rho_state_dict': model_list[1].mean_module_NN_All.state_dict()}, save_folder + f'Trained_models_{epoch}.pth')
                total_time = total_time + (end_time - start_time)
                start_time = time.time()

            model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
            model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
            model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])
            model_list[0].MP['s'] = min(model_list[0].MP['s_f'], model_list[0].MP['s'] + model_list[0].MP['frac_step_source'])


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

        torch.save({
            "T_X_train": T_X_train, "T_train": T_train,
            "rho_X_train": rho_X_train, "rho_train": rho_train
        }, save_folder + "Training.pth")

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
    base_folder = f"/home/alexsunuci/PIGP2D_Dec20th_2025/"
    # base_folder = f'C:/Users/Alex/Desktop/Heat sink/PIGP2D_Dec20th_2025/'
    random_state = [1,3,5,7,9,11,13,15,17,19]
    # random_state = [1,3,5]
    num_CP = 51
    Example = 'EX2D1'
    Case = 'single_material_2'
    Diff_type = 'CPS4_reduced'
    N_worker = 1  # for local debugging with 1 GPU
    # define material properties and
    # wt = [1.0, 1.0, 1.0, 1.0] # full integration
    wt = [4.0] # reduced integration
    D = [0, 1.0]  # density
    kappa = [1e-5, 1.0]  # conductivity
    E = [1e-5, 1.0]  # modulus
    P = [0, 1.0]  # cost
    source = 1e-5 # heat sink
    source_f = 2e-5 # final value of heat sink
    nu = 0.3  # Poisson's ratio
    p0 = 3  # initial penalty
    pf = 3  # final penalty
    massfrac_f = 0.3  # final mass fraction
    costfrac_f = 0.3  # final cost fraction
    frac_decrease = 0.5  # fraction of epoch to decrease volume fraction from VF0 to VF_f
    b = 8  # sharpness parameters
    thres_static = 0.5  # static threshold values to binarize the density field
    rho_min = 0.1  # lower limit to define grey element (rho_min,rho_max)
    rho_max = 0.9  # upper limit to define grey element (rho_min,rho_max)

    # define domain
    Nelx = 200
    Nely = 100
    xmin, xmax = 0.0, 200.0
    ymin, ymax = 0.0, 100.0

    # define model parameters and training parameters
    init_method = 'kaiming_uniform_'
    dynamic_weight = False
    gradient_clip = False
    omega = 0.5
    learning_rate_disp = 1e-3
    learning_rate_rho = 1e-4
    nrmThreshold = 0.1
    TO_num_iter = 10000
    plot_num = 10
    delta = 1e-1
    wc = 1
    wd = 1e3
    wm = 1e3
    wp = 1e3
    basis = 'PGCAN'
    basis_rho = 'PGCAN'
    quant_correlation_class = 'Rough_RBF'
    activation = 'tanh'
    if basis == 'PGCAN':
        n_features = 128
        n_cells = 3
        res = [36, 72]
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
    kappa_tensor = [torch.tensor(kappa, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    wt_tensor = [torch.tensor(wt, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    domain = {'x': [xmin, xmax], 'y': [ymin, ymax]}
    frac_decrease = int(frac_decrease * TO_num_iter)
    plotting_interval_TO = TO_num_iter // plot_num
    frac_step = None
    frac_step_source = (source_f - source) / frac_decrease
    MP = {'num_CP': num_CP, 'Example': Example, 'N_worker': N_worker,
          'base_folder': base_folder, 'domain': domain,
          'thres_static': thres_static, 'Diff_type': Diff_type,
          'num_phase': num_phase, 'Nelx': Nelx, 'Nely': Nely, 'kappa': kappa_tensor,
          'D': D_tensor, 'E': E_tensor, 'P': P_tensor, 'wt':wt_tensor,'nu': nu,
          'p': p0, 'p0': p0, 'pf': pf, 'b': b,'s':source,'s_f':source_f,
          'rho_min': rho_min, 'rho_max': rho_max,
          'massfrac_star': massfrac_f, 'massfrac0': massfrac_f,
          'massfrac_f': massfrac_f, 'frac_decrease': frac_decrease,
          'costfrac_star': costfrac_f, 'costfrac0': costfrac_f,
          'costfrac_f': costfrac_f, 
          'frac_step_mass': frac_step, 'frac_step_cost': frac_step,
          'frac_step_p': frac_step,'frac_step_source': frac_step_source}

    NN_config_disp = {'init_method': init_method, 'random_state': random_state, 'state': [],
                      'Example': Example, 'Case': Case,
                      'dynamic_weight': dynamic_weight, 'gradient_clip': gradient_clip,
                      'omega': omega, 'learning_rate_disp': learning_rate_disp,
                      'learning_rate_rho': learning_rate_rho, 'nrmThreshold': nrmThreshold,
                      'TO_num_iter': TO_num_iter, 'plot_num': plot_num,
                      'plotting_interval_TO': plotting_interval_TO,
                      'delta': delta, 'wc': wc, 'wd': wd, 'wm': wm, 'wp': wp,
                      'kernel_size': kernel_size,
                      'basis': basis, 'quant_correlation_class': quant_correlation_class,
                      'activation': activation,
                      'n_features': n_features, 'n_cells': n_cells,
                      'res': res, 'NN_arch': NN_arch, 'save_folder': []}

    NN_config_rho = copy.deepcopy(NN_config_disp)
    NN_config_rho['basis'] = basis_rho
    if basis_rho == 'PGCAN':
        NN_config_rho['n_features'] = 128
        NN_config_rho['n_cells'] = 6
        NN_config_rho['res'] = [36, 72]
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
    T_X_train = Training['T_X_train']
    rho_X_train = Training['rho_X_train']
    T_train = Training['T_train']
    rho_train = Training['rho_train']

    # Generate random integers
    index_CP = np.random.randint(1, num_CP + 1, size=TO_num_iter)

    # Call worker directly (torchrun handles ranks)
    run_worker(MP, NN_config_disp, NN_config_rho,
               T_X_train, T_train,
               rho_X_train, rho_train,
               index_CP, mesh_data)


