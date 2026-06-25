import torch
import numpy as np
from utils.utils_general import get_multiGPU
import matplotlib.pyplot as plt

def get_data_EX2D1(MP):
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
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)
    l = ymax - ymin
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    index = (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymin + l/20) & (X_col[:, 0] == xmin)
    LB = X_col[index] # left edge coordinates

    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(LB[::1,0:1], LB[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    T_X_train = torch.tensor(LB).type(tkwargs["dtype"]).requires_grad_(False)
    T_train = (torch.zeros_like(T_X_train)[:,0]).type(tkwargs["dtype"])

    rho_X_train = torch.tensor([[-1.0,-1.0]]).type(tkwargs["dtype"]).requires_grad_(False)
    rho_train = torch.tensor([0.0]).type(tkwargs["dtype"])

    Training = {'T_X_train':T_X_train,'T_train':T_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train}
    
    # process the training data
    Training_multi = {}
    for i, device in enumerate(tkwargs["devices"]):
        key = f"GPU{i}"
        Training_multi[key] = {
            "T_X_train": Training["T_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "rho_X_train": Training["rho_X_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "T_train": Training["T_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
            "rho_train": Training["rho_train"].type(tkwargs["dtype"]).to(device).requires_grad_(False),
        }
    
    # process the mesh data
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    # float + gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "conn"]:
                    # indices must stay long
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    # float but no gradients
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)

    return Training, mesh_data









