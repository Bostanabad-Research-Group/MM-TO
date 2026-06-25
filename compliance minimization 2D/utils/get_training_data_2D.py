import torch
import numpy as np
from utils.utils_general import get_tkwargs
import matplotlib.pyplot as plt

tkwargs = get_tkwargs()

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

def get_data_EX2D1(MP):
    # get collocation points
    pad = MP['pad']
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)
    domain_volume = (xmax - xmin)*(ymax - ymin)
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # delete points in unwanted area:
    distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
    mask_hole = (distance < hole_radius)
    mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
    mask_col = mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)

    index = (X_col[:, 0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates
    index = (X_col[:, 0] == xmax) & (X_col[:, 1] == 0)
    Bottom_support = X_col[index] 

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

    # define the traction BC:
    traction_coord = torch.tensor([xmin, ymax])
    traction_magnitude = torch.tensor(-1e-1)
    traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax)

    masked_domain = X_col[mask_col]
    # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.3, s=3, color='purple', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(traction_coord[0], traction_coord[1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')
    # ax.scatter(Bottom_support[0,0], Bottom_support[0,1], marker='o', alpha=0.9, s=2, color='black', label = 'bottom support')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    # traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    # traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(Bottom_support).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = torch.vstack([traction_coord,Bottom_support]).detach()
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi = np.linspace(xmin, xmax, 3*Nelx), np.linspace(ymin, ymax, 3*Nely)
    xi, yi = np.meshgrid(xi, yi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'LE':LE,'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny,
                'mask_nan':mask_nan,'xi':xi,'yi':yi,'domain_volume':domain_volume,'detJ':detJ,'dN':dN,
                'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem, 'xi_elem':xi_elem,'yi_elem':yi_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        xi, yi = np.meshgrid(xi, yi)

        # delete points in unwanted area:
        distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
        mask_hole = (distance < hole_radius)
        mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
        mask_col = mask_domain
        Nx = mask_col.shape[1]
        Ny = mask_col.shape[0]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax)
        # traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])
        traction_magnitude = torch.tensor(-1e-1)

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
        # mask_col_elem = mask_domain_elem.flatten()
        # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten()]).T)
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
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'xi':xi,'yi':yi, 
        'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'xi_elem':xi_elem,'yi_elem':yi_elem,'detJ':detJ, 'dN':dN}

        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX2D2(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)
    domain_volume = (xmax - xmin)*(ymax - ymin)
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # delete points in unwanted area:
    distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
    mask_hole = (distance < hole_radius)
    mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
    mask_col = mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
    
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

    index = (X_col[:, 0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates

    # define the traction BC:
    traction_coord = torch.tensor([[xmax, ymin]])
    traction_magnitude = torch.tensor(-1e-1)
    traction_indices = (X_col[:, 0] == xmax) & (X_col[:, 1] == ymin)
    
    # # Visualize ALL the training points
    # masked_domain = X_col[mask_col]
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.3, s=3, color='purple', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(traction_coord[0,0], traction_coord[0,1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi = np.linspace(xmin, xmax, 3*Nelx), np.linspace(ymin, ymax, 3*Nely)
    xi, yi = np.meshgrid(xi, yi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'LE':LE,'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny,
                'mask_nan':mask_nan,'xi':xi,'yi':yi,'domain_volume':domain_volume,'detJ':detJ,'dN':dN,
                'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem, 'xi_elem':xi_elem,'yi_elem':yi_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        xi, yi = np.meshgrid(xi, yi)

        # delete points in unwanted area:
        distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
        mask_hole = (distance < hole_radius)
        mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
        mask_col = mask_domain
        Nx = mask_col.shape[1]
        Ny = mask_col.shape[0]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_coord = torch.tensor([[xmax, ymin]])
        traction_magnitude = torch.tensor(-1e-1)
        traction_indices = (X_col[:, 0] == xmax) & (X_col[:, 1] == ymin)
        traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
        # traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

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
        # mask_col_elem = mask_domain_elem.flatten()
        # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten()]).T)
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
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'xi':xi,'yi':yi, 
        'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'xi_elem':xi_elem,'yi_elem':yi_elem,'detJ':detJ, 'dN':dN}

        # Append the dictionary to the list
        X_col_all.append(X_col_data)

    return Training, X_col_all

def get_data_EX2D3(MP):
    # get collocation points
    pad = MP['pad']
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)
    domain_volume = (xmax - xmin)*(ymax - ymin)
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # delete points in unwanted area:
    distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
    mask_hole = (distance < hole_radius)
    mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
    mask_col = mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)

    index = (X_col[:, 0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax)
    LE = X_col[index] # left edge coordinates
    index = (X_col[:, 0] == xmax) & (X_col[:, 1] == 0)
    Bottom_support = X_col[index] 

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

    # define the traction BC:
    traction_coord = torch.tensor([xmin, ymax])
    traction_magnitude1 = torch.tensor([-1e-1])  # make it a 1D tensor for concatenation
    traction_indices1_mask = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymin)
    traction_indices1 = torch.nonzero(traction_indices1_mask, as_tuple=False).squeeze()

    center = (xmax + xmin) / 2  # x-coordinate of center
    segment = 2
    traction_indices2_mask = (
        (X_col[:, 0] >= (center - segment / 2)) &
        (X_col[:, 0] <= (center + segment / 2)) &
        (X_col[:, 1] == ymin)
    )
    traction_indices2 = torch.nonzero(traction_indices2_mask, as_tuple=False).squeeze()
    num_points = len(traction_indices2)
    total_force = -1e-1
    traction_magnitude2 = (total_force / num_points) * torch.ones_like(traction_indices2, dtype=torch.float)

    # Concatenate indices and magnitudes
    traction_indices = torch.cat((traction_indices1.unsqueeze(0), traction_indices2))
    traction_magnitude = torch.cat((traction_magnitude1, traction_magnitude2))

    masked_domain = X_col[mask_col]
    # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='blue', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.3, s=3, color='purple', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    # ax.scatter(traction_coord[0], traction_coord[1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')
    # ax.scatter(Bottom_support[0,0], Bottom_support[0,1], marker='o', alpha=0.9, s=2, color='black', label = 'bottom support')
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # define the training dataset
    # traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    # traction_indices = traction_indices.requires_grad_(False).to(tkwargs['device'])

    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(Bottom_support).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = torch.vstack([traction_coord,Bottom_support]).detach()
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi = np.linspace(xmin, xmax, 3*Nelx), np.linspace(ymin, ymax, 3*Nely)
    xi, yi = np.meshgrid(xi, yi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'LE':LE,'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny,
                'mask_nan':mask_nan,'xi':xi,'yi':yi,'domain_volume':domain_volume,'detJ':detJ,'dN':dN,
                'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem, 'xi_elem':xi_elem,'yi_elem':yi_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        xi, yi = np.meshgrid(xi, yi)

        # delete points in unwanted area:
        distance = ((xi - hole_center[0]) ** 2 + (yi- hole_center[1]) ** 2) ** 0.5
        mask_hole = (distance < hole_radius)
        mask_domain = (distance >= hole_radius) # mask domain contains hole points, use GP on those points for density
        mask_col = mask_domain
        Nx = mask_col.shape[1]
        Ny = mask_col.shape[0]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_coord = torch.tensor([xmin, ymax])
        traction_magnitude1 = torch.tensor([-1e-1]).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])  # make it a 1D tensor for concatenation
        traction_indices1_mask = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymin)
        traction_indices1 = torch.nonzero(traction_indices1_mask, as_tuple=False).squeeze()

        center = (xmax + xmin) / 2  # x-coordinate of center
        segment = 2
        traction_indices2_mask = (
            (X_col[:, 0] >= (center - segment / 2)) &
            (X_col[:, 0] <= (center + segment / 2)) &
            (X_col[:, 1] == ymin)
        )
        traction_indices2 = torch.nonzero(traction_indices2_mask, as_tuple=False).squeeze()
        num_points = len(traction_indices2)
        total_force = -1e-1
        traction_magnitude2 = (total_force / num_points) * torch.ones_like(traction_indices2, dtype=torch.float)

        # Concatenate indices and magnitudes
        traction_indices = torch.cat((traction_indices1.unsqueeze(0), traction_indices2))
        traction_magnitude = torch.cat((traction_magnitude1, traction_magnitude2))
        traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

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
        # mask_col_elem = mask_domain_elem.flatten()
        # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten()]).T)
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
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'xi':xi,'yi':yi, 
        'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'xi_elem':xi_elem,'yi_elem':yi_elem,'detJ':detJ, 'dN':dN}

        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX2D4(MP):
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    xi, yi = np.meshgrid(xi, yi)
    domain_volume = (xmax - xmin)*(ymax - ymin) - 40*40
    dxds=dx/2
    dydt=dy/2
    J= np.array([[dxds,0],[0,dydt]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_2D(Jinv,MP['Diff_type'])

    # delete points in unwanted area:
    mask_hole = (xi>40.0) & (yi>40.0)
    mask_domain = (xi<=40.0) | (yi<=40.0)
    mask_col = mask_domain
    Nx = mask_col.shape[1]
    Ny = mask_col.shape[0]
    mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
    mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)

    # create meshgrid for fem element center
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    xi, yi = np.meshgrid(xi, yi)
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1]
    distance = ((xi_elem - hole_center[0]) ** 2 + (yi_elem- hole_center[1]) ** 2) ** 0.5
    mask_hole_elem = (xi_elem > 40.0) & (yi_elem > 40.0)
    mask_domain_elem = (xi_elem <= 40.0) | (yi_elem <= 40.0) # mask domain contains hole points, use GP on those points 
    mask_col_elem = mask_domain_elem.T.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.T.flatten(),yi_elem.T.flatten()]).T)

    index = (X_col[:, 1] == ymax) & (X_col[:, 0] >= xmin) & (X_col[:, 0] <= 40.0)
    TE = X_col[index] # left edge coordinates

    # define the traction BC:
    traction_coord = torch.tensor([[xmax, 40.0]])
    traction_magnitude = torch.tensor(-1e-1)
    masked_domain = X_col[mask_col]
    distances = torch.norm(masked_domain - traction_coord, dim=1)
    closest_point_idx_in_mask = torch.argmin(distances).item()
    Coord_masked = masked_domain[closest_point_idx_in_mask]
    traction_indices = (X_col[:, 0] == Coord_masked[0]) & (X_col[:, 1] == Coord_masked[1])

    # Visualize ALL the training points
    step = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_col[::step,0:1], X_col[::step,1:2], marker='o', alpha=0.3, s=3, color='cyan', label = 'All grid points')
    ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2], marker='o', alpha=0.9, s=3, color='blue', label = 'masked domain')
    ax.scatter(TE[::1,0:1], TE[::1,1:2], marker='o', alpha=0.9, s=2, color='red', label = 'left edge')
    ax.scatter(X_col[traction_indices,0], X_col[traction_indices,1], marker='o', alpha=0.9, s=2, color='green', label = 'external force')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

    # define the training dataset
    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    TE = torch.tensor(TE).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    u_X_train = torch.tensor(TE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(TE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord
    rho_X_train = rho_X_train.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi = np.linspace(xmin, xmax, 2*Nelx), np.linspace(ymin, ymax, 2*Nely)
    xi, yi = np.meshgrid(xi, yi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2)
    mask_nan = (xi > 40) & (yi > 40) # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_coord':traction_coord,
                'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'TE':TE,'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'Nx':Nx,'Ny':Ny,
                'mask_nan':mask_nan,'xi':xi,'yi':yi,'domain_volume':domain_volume,'detJ':detJ,'dN':dN,
                'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem, 'xi_elem':xi_elem,'yi_elem':yi_elem}

    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        xi, yi = np.meshgrid(xi, yi)

        # delete points in unwanted area:
        mask_domain = (xi<=40.0) | (yi<=40.0) # no holes, so all True
        mask_hole = (xi>40.0) & (yi>40.0)
        mask_col = mask_domain
        Nx = mask_col.shape[1]
        Ny = mask_col.shape[0]
        mask_col = mask_col.T.flatten() # reshape it to the same size with X_col
        mask_hole = mask_hole.T.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.T.flatten(),yi.T.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # define the traction BC:
        traction_coord = torch.tensor([[xmax, 40.0]]).to(tkwargs['device'])
        traction_magnitude = torch.tensor(-1e-1)
        masked_domain = X_col[mask_col]
        distances = torch.norm(masked_domain - traction_coord, dim=1)
        closest_point_idx_in_mask = torch.argmin(distances).item()
        Coord_masked = masked_domain[closest_point_idx_in_mask]
        traction_indices = (X_col[:, 0] == Coord_masked[0]) & (X_col[:, 1] == Coord_masked[1])

        # create meshgrid for fem element center
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        xi, yi = np.meshgrid(xi, yi)
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1]
        mask_domain_elem = (xi_elem<=40.0) | (yi_elem<=40.0) # no holes, so all True
        mask_hole_elem = (xi_elem>40.0) & (yi_elem>40.0)
        # mask_col_elem = mask_domain_elem.flatten()
        # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten()]).T)
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
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'dx':dx,'dy':dy,'xi':xi,'yi':yi, 
        'mask_col_elem':mask_col_elem,'X_col_elem':X_col_elem,'xi_elem':xi_elem,'yi_elem':yi_elem,'detJ':detJ, 'dN':dN}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

