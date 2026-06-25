import torch
import numpy as np
from utils.utils_general import get_multiGPU
import matplotlib.pyplot as plt

def get_data_EX2D2_four_phase_rho_block(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    
    tkwargs = get_multiGPU(N_worker)
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 200) | (yi <=100)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymin + 4.0)
    LB = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= xmax)
    tp = X_col[index] # left edge coordinates

    # define the gripper contact
    edge_length = 3.0
    left_block = (
        (X_col[:, 0] >= 0.0) &
        (X_col[:, 0] <= edge_length) &
        (X_col[:, 1] >= ymax - edge_length) &
        (X_col[:, 1] <= ymax)
    )
    right_block = (
        (X_col[:, 0] >= xmax - edge_length) &
        (X_col[:, 0] <= xmax) &
        (X_col[:, 1] >= ymax - edge_length) &
        (X_col[:, 1] <= ymax)
    )
    index = left_block | right_block

    gripper_contact = X_col[index]

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LB[::1,0:1], LB[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left bottom')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')
    # ax.scatter(gripper_contact[::1,0:1], gripper_contact[::1,1:2], marker='o', alpha=0.9, s=2, color='black', label = 'gripper contact')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    u_X_train = torch.tensor(LB).type(tkwargs["dtype"]).requires_grad_(False)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    LB = torch.tensor(LB)
    tp = torch.tensor(tp)
    v_X_train = torch.cat([LB,tp]).type(tkwargs["dtype"]).requires_grad_(False)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    t1_X_train = u_X_train
    t1_train = u_train
    t2_X_train = v_X_train
    t2_train = v_train

    phase0_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase2_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase3_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase2_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase3_train = (torch.ones_like(gripper_contact)[:,0]).type(tkwargs["dtype"])


    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                't1_X_train':t1_X_train,'t1_train':t1_train,'t2_X_train':t2_X_train,'t2_train':t2_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,
                'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,
                'phase2_X_train':phase2_X_train,'phase2_train':phase2_train,
                'phase3_X_train':phase3_X_train,'phase3_train':phase3_train,
                'LB':LB,}
    
    # process the training data
    Training_multi = {}
    for i, device in enumerate(tkwargs["devices"]):
        key = f"GPU{i}"
        Training_multi[key] = {
            "u_X_train": Training["u_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_X_train": Training["v_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_X_train": Training["t1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_X_train": Training["t2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "u_train": Training["u_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_train": Training["v_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_train": Training["t1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_train": Training["t2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_X_train": Training["phase0_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_X_train": Training["phase1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase2_X_train": Training["phase2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase3_X_train": Training["phase3_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_train": Training["phase0_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_train": Training["phase1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase2_train": Training["phase2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase3_train": Training["phase3_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
        }

    
    # process the mesh data
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    # float + gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    # indices must stay long
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    # float but no gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data

def get_data_EX2D2_two_phase_rho_block(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    
    tkwargs = get_multiGPU(N_worker)
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 200) | (yi <=100)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymin + 4.0)
    LB = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= xmax)
    tp = X_col[index] # left edge coordinates

    # define the gripper contact
    edge_length = 3.0
    left_block = (
        (X_col[:, 0] >= 0.0) &
        (X_col[:, 0] <= edge_length) &
        (X_col[:, 1] >= ymax - edge_length) &
        (X_col[:, 1] <= ymax)
    )
    right_block = (
        (X_col[:, 0] >= xmax - edge_length) &
        (X_col[:, 0] <= xmax) &
        (X_col[:, 1] >= ymax - edge_length) &
        (X_col[:, 1] <= ymax)
    )
    index = left_block | right_block

    gripper_contact = X_col[index]

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LB[::1,0:1], LB[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left bottom')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')
    # ax.scatter(gripper_contact[::1,0:1], gripper_contact[::1,1:2], marker='o', alpha=0.9, s=2, color='black', label = 'gripper contact')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    u_X_train = torch.tensor(LB).type(tkwargs["dtype"]).requires_grad_(False)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    LB = torch.tensor(LB)
    tp = torch.tensor(tp)
    v_X_train = torch.cat([LB,tp]).type(tkwargs["dtype"]).requires_grad_(False)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    t1_X_train = u_X_train
    t1_train = u_train
    t2_X_train = v_X_train
    t2_train = v_train

    phase0_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.ones_like(gripper_contact)[:,0]).type(tkwargs["dtype"])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                't1_X_train':t1_X_train,'t1_train':t1_train,'t2_X_train':t2_X_train,'t2_train':t2_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,'LB':LB,}
    
    # process the training data
    Training_multi = {}
    for i, device in enumerate(tkwargs["devices"]):
        key = f"GPU{i}"
        Training_multi[key] = {
            "u_X_train": Training["u_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_X_train": Training["v_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_X_train": Training["t1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_X_train": Training["t2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "u_train": Training["u_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_train": Training["v_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_train": Training["t1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_train": Training["t2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_X_train": Training["phase0_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_X_train": Training["phase1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_train": Training["phase0_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_train": Training["phase1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
        }
    
    # process the mesh data
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    # float + gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    # indices must stay long
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    # float but no gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data

def get_data_EX2D1_gripper_four_phase(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    
    tkwargs = get_multiGPU(N_worker)
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 160) | (yi <=60)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymin + 4.0)
    LB = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= 160.0)
    tp = X_col[index] # left edge coordinates

    # define the gripper contact
    index = (X_col[:, 0] >= 160.0) & (X_col[:, 0] <= xmax) & (X_col[:, 1] >= 55.0) & (X_col[:, 1] <= 60.0)
    gripper_contact = X_col[index]

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LB[::1,0:1], LB[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left bottom')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')
    # ax.scatter(gripper_contact[::1,0:1], gripper_contact[::1,1:2], marker='o', alpha=0.9, s=2, color='black', label = 'gripper contact')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    u_X_train = torch.tensor(LB).type(tkwargs["dtype"]).requires_grad_(False)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    LB = torch.tensor(LB)
    tp = torch.tensor(tp)
    v_X_train = torch.cat([LB,tp]).type(tkwargs["dtype"]).requires_grad_(False)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    t1_X_train = u_X_train
    t1_train = u_train
    t2_X_train = v_X_train
    t2_train = v_train

    phase0_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase2_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase3_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase2_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase3_train = (torch.ones_like(gripper_contact)[:,0]).type(tkwargs["dtype"])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                't1_X_train':t1_X_train,'t1_train':t1_train,'t2_X_train':t2_X_train,'t2_train':t2_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,
                'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,
                'phase2_X_train':phase2_X_train,'phase2_train':phase2_train,
                'phase3_X_train':phase3_X_train,'phase3_train':phase3_train,
                'LB':LB,}
    
    # process the training data
    Training_multi = {}
    for i, device in enumerate(tkwargs["devices"]):
        key = f"GPU{i}"
        Training_multi[key] = {
            "u_X_train": Training["u_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_X_train": Training["v_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_X_train": Training["t1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_X_train": Training["t2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_X_train": Training["phase0_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_X_train": Training["phase1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase2_X_train": Training["phase2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase3_X_train": Training["phase3_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "u_train": Training["u_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_train": Training["v_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_train": Training["t1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_train": Training["t2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_train": Training["phase0_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_train": Training["phase1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase2_train": Training["phase2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase3_train": Training["phase3_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
        }
    
    # process the mesh data
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    # float + gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    # indices must stay long
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    # float but no gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data

def get_data_EX2D1_gripper_two_phase(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + 'Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    
    tkwargs = get_multiGPU(N_worker)
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    mask_col = (xi <= 160) | (yi <=60)
    mask_col = mask_col.T.flatten()
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    X_col = X_col[mask_col]
    index = (X_col[:, 0] == xmin) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymin + 4.0)
    LB = X_col[index] # left edge coordinates

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= 160.0)
    tp = X_col[index] # left edge coordinates

    # define the gripper contact
    index = (X_col[:, 0] >= 160.0) & (X_col[:, 0] <= xmax) & (X_col[:, 1] >= 55.0) & (X_col[:, 1] <= 60.0)
    gripper_contact = X_col[index]

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LB[::1,0:1], LB[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left bottom')
    # ax.scatter(tp[::1,0:1], tp[::1,1:2], marker='o', alpha=0.9, s=2, color='cyan', label = 'top edge')
    # ax.scatter(gripper_contact[::1,0:1], gripper_contact[::1,1:2], marker='o', alpha=0.9, s=2, color='black', label = 'gripper contact')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    u_X_train = torch.tensor(LB).type(tkwargs["dtype"]).requires_grad_(False)
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"])

    LB = torch.tensor(LB)
    tp = torch.tensor(tp)
    v_X_train = torch.cat([LB,tp]).type(tkwargs["dtype"]).requires_grad_(False)
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"])

    t1_X_train = u_X_train
    t1_train = u_train
    t2_X_train = v_X_train
    t2_train = v_train

    phase0_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase1_X_train = torch.tensor(gripper_contact).type(tkwargs["dtype"]).requires_grad_(False)
    phase0_train = (torch.zeros_like(gripper_contact)[:,0]).type(tkwargs["dtype"])
    phase1_train = (torch.ones_like(gripper_contact)[:,0]).type(tkwargs["dtype"])

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                't1_X_train':t1_X_train,'t1_train':t1_train,'t2_X_train':t2_X_train,'t2_train':t2_train,
                'phase0_X_train':phase0_X_train,'phase0_train':phase0_train,'phase1_X_train':phase1_X_train,'phase1_train':phase1_train,'LB':LB,}
    
    # process the training data
    Training_multi = {}
    for i, device in enumerate(tkwargs["devices"]):
        key = f"GPU{i}"
        Training_multi[key] = {
            "u_X_train": Training["u_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_X_train": Training["v_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_X_train": Training["t1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_X_train": Training["t2_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_X_train": Training["phase0_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_X_train": Training["phase1_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "u_train": Training["u_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "v_train": Training["v_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t1_train": Training["t1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "t2_train": Training["t2_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase0_train": Training["phase0_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "phase1_train": Training["phase1_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
        }
    
    # process the mesh data
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    # float + gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    # indices must stay long
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    # float but no gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data
