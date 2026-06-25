import torch
import numpy as np
from utils.utils_general import get_tkwargs
# import matplotlib.pyplot as plt

tkwargs = get_tkwargs()

def get_dN_3D(Diff_type,Jinv):
    if Diff_type == 'hex':
        ## Differentiation of shape functions at gauss quadrature points       
        dN1_dsy=np.array([[-0.311004233964073, -0.311004233964073, -0.0833333333333333, -0.0833333333333333, -0.0833333333333333, -0.0833333333333333, -0.0223290993692602, -0.0223290993692602],
                          [-0.311004233964073, -0.0833333333333333, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0223290993692602, -0.0833333333333333],
                          [-0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333]])
        dN2_dsy=np.array([[0.311004233964073, 0.311004233964073, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0223290993692602, 0.0223290993692602],
                          [-0.0833333333333333, -0.311004233964073, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.0833333333333333, -0.0223290993692602],
                          [-0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602]])
        dN3_dsy=np.array([[0.0833333333333333, 0.0833333333333333, 0.311004233964073, 0.311004233964073, 0.0223290993692602, 0.0223290993692602, 0.0833333333333333, 0.0833333333333333],
                          [0.0833333333333333, 0.311004233964073, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.0833333333333333, 0.0223290993692602],
                          [-0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333]])
        dN4_dsy=np.array([[-0.0833333333333333, -0.0833333333333333, -0.311004233964073, -0.311004233964073, -0.0223290993692602, -0.0223290993692602, -0.0833333333333333, -0.0833333333333333],
                          [0.311004233964073, 0.0833333333333333, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0223290993692602, 0.0833333333333333],
                          [-0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073]])
        dN5_dsy=np.array([[-0.0833333333333333, -0.0833333333333333, -0.0223290993692602, -0.0223290993692602, -0.311004233964073, -0.311004233964073, -0.0833333333333333, -0.0833333333333333],
                          [-0.0833333333333333, -0.0223290993692602, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.0833333333333333, -0.0833333333333333, -0.311004233964073],
                          [0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333]])
        dN6_dsy=np.array([[0.0833333333333333, 0.0833333333333333, 0.0223290993692602, 0.0223290993692602, 0.311004233964073, 0.311004233964073, 0.0833333333333333, 0.0833333333333333],
                          [-0.0223290993692602, -0.0833333333333333, -0.0833333333333333, -0.0223290993692602, -0.0833333333333333, -0.311004233964073, -0.311004233964073, -0.0833333333333333],
                          [0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602]])
        dN7_dsy=np.array([[0.0223290993692602, 0.0223290993692602, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.0833333333333333, 0.311004233964073, 0.311004233964073],
                          [0.0223290993692602, 0.0833333333333333, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.311004233964073, 0.0833333333333333],
                          [0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333]])
        dN8_dsy=np.array([[-0.0223290993692602, -0.0223290993692602, -0.0833333333333333, -0.0833333333333333, -0.0833333333333333, -0.0833333333333333, -0.311004233964073, -0.311004233964073],
                          [0.0833333333333333, 0.0223290993692602, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0833333333333333, 0.311004233964073],
                          [0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073, 0.0833333333333333, 0.0223290993692602, 0.0833333333333333, 0.311004233964073]])

        dN1_GP1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0],dN1_dsy[2][0]]).reshape((3,1)))
        dN2_GP1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0],dN2_dsy[2][0]]).reshape((3,1)))
        dN3_GP1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0],dN3_dsy[2][0]]).reshape((3,1)))
        dN4_GP1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0],dN4_dsy[2][0]]).reshape((3,1)))
        dN5_GP1=np.matmul(Jinv, np.array([dN5_dsy[0][0],dN5_dsy[1][0],dN5_dsy[2][0]]).reshape((3,1)))
        dN6_GP1=np.matmul(Jinv, np.array([dN6_dsy[0][0],dN6_dsy[1][0],dN6_dsy[2][0]]).reshape((3,1)))
        dN7_GP1=np.matmul(Jinv, np.array([dN7_dsy[0][0],dN7_dsy[1][0],dN7_dsy[2][0]]).reshape((3,1)))
        dN8_GP1=np.matmul(Jinv, np.array([dN8_dsy[0][0],dN8_dsy[1][0],dN8_dsy[2][0]]).reshape((3,1)))

        dN1_GP2=np.matmul(Jinv, np.array([dN1_dsy[0][1],dN1_dsy[1][1],dN1_dsy[2][1]]).reshape((3,1)))
        dN2_GP2=np.matmul(Jinv, np.array([dN2_dsy[0][1],dN2_dsy[1][1],dN2_dsy[2][1]]).reshape((3,1)))
        dN3_GP2=np.matmul(Jinv, np.array([dN3_dsy[0][1],dN3_dsy[1][1],dN3_dsy[2][1]]).reshape((3,1)))
        dN4_GP2=np.matmul(Jinv, np.array([dN4_dsy[0][1],dN4_dsy[1][1],dN4_dsy[2][1]]).reshape((3,1)))
        dN5_GP2=np.matmul(Jinv, np.array([dN5_dsy[0][1],dN5_dsy[1][1],dN5_dsy[2][1]]).reshape((3,1)))
        dN6_GP2=np.matmul(Jinv, np.array([dN6_dsy[0][1],dN6_dsy[1][1],dN6_dsy[2][1]]).reshape((3,1)))
        dN7_GP2=np.matmul(Jinv, np.array([dN7_dsy[0][1],dN7_dsy[1][1],dN7_dsy[2][1]]).reshape((3,1)))
        dN8_GP2=np.matmul(Jinv, np.array([dN8_dsy[0][1],dN8_dsy[1][1],dN8_dsy[2][1]]).reshape((3,1)))

        dN1_GP3=np.matmul(Jinv, np.array([dN1_dsy[0][2],dN1_dsy[1][2],dN1_dsy[2][2]]).reshape((3,1)))
        dN2_GP3=np.matmul(Jinv, np.array([dN2_dsy[0][2],dN2_dsy[1][2],dN2_dsy[2][2]]).reshape((3,1)))
        dN3_GP3=np.matmul(Jinv, np.array([dN3_dsy[0][2],dN3_dsy[1][2],dN3_dsy[2][2]]).reshape((3,1)))
        dN4_GP3=np.matmul(Jinv, np.array([dN4_dsy[0][2],dN4_dsy[1][2],dN4_dsy[2][2]]).reshape((3,1)))
        dN5_GP3=np.matmul(Jinv, np.array([dN5_dsy[0][2],dN5_dsy[1][2],dN5_dsy[2][2]]).reshape((3,1)))
        dN6_GP3=np.matmul(Jinv, np.array([dN6_dsy[0][2],dN6_dsy[1][2],dN6_dsy[2][2]]).reshape((3,1)))
        dN7_GP3=np.matmul(Jinv, np.array([dN7_dsy[0][2],dN7_dsy[1][2],dN7_dsy[2][2]]).reshape((3,1)))
        dN8_GP3=np.matmul(Jinv, np.array([dN8_dsy[0][2],dN8_dsy[1][2],dN8_dsy[2][2]]).reshape((3,1)))

        dN1_GP4=np.matmul(Jinv, np.array([dN1_dsy[0][3],dN1_dsy[1][3],dN1_dsy[2][3]]).reshape((3,1)))
        dN2_GP4=np.matmul(Jinv, np.array([dN2_dsy[0][3],dN2_dsy[1][3],dN2_dsy[2][3]]).reshape((3,1)))
        dN3_GP4=np.matmul(Jinv, np.array([dN3_dsy[0][3],dN3_dsy[1][3],dN3_dsy[2][3]]).reshape((3,1)))
        dN4_GP4=np.matmul(Jinv, np.array([dN4_dsy[0][3],dN4_dsy[1][3],dN4_dsy[2][3]]).reshape((3,1)))
        dN5_GP4=np.matmul(Jinv, np.array([dN5_dsy[0][3],dN5_dsy[1][3],dN5_dsy[2][3]]).reshape((3,1)))
        dN6_GP4=np.matmul(Jinv, np.array([dN6_dsy[0][3],dN6_dsy[1][3],dN6_dsy[2][3]]).reshape((3,1)))
        dN7_GP4=np.matmul(Jinv, np.array([dN7_dsy[0][3],dN7_dsy[1][3],dN7_dsy[2][3]]).reshape((3,1)))
        dN8_GP4=np.matmul(Jinv, np.array([dN8_dsy[0][3],dN8_dsy[1][3],dN8_dsy[2][3]]).reshape((3,1)))

        dN1_GP5=np.matmul(Jinv, np.array([dN1_dsy[0][4],dN1_dsy[1][4],dN1_dsy[2][4]]).reshape((3,1)))
        dN2_GP5=np.matmul(Jinv, np.array([dN2_dsy[0][4],dN2_dsy[1][4],dN2_dsy[2][4]]).reshape((3,1)))
        dN3_GP5=np.matmul(Jinv, np.array([dN3_dsy[0][4],dN3_dsy[1][4],dN3_dsy[2][4]]).reshape((3,1)))
        dN4_GP5=np.matmul(Jinv, np.array([dN4_dsy[0][4],dN4_dsy[1][4],dN4_dsy[2][4]]).reshape((3,1)))
        dN5_GP5=np.matmul(Jinv, np.array([dN5_dsy[0][4],dN5_dsy[1][4],dN5_dsy[2][4]]).reshape((3,1)))
        dN6_GP5=np.matmul(Jinv, np.array([dN6_dsy[0][4],dN6_dsy[1][4],dN6_dsy[2][4]]).reshape((3,1)))
        dN7_GP5=np.matmul(Jinv, np.array([dN7_dsy[0][4],dN7_dsy[1][4],dN7_dsy[2][4]]).reshape((3,1)))
        dN8_GP5=np.matmul(Jinv, np.array([dN8_dsy[0][4],dN8_dsy[1][4],dN8_dsy[2][4]]).reshape((3,1)))

        dN1_GP6=np.matmul(Jinv, np.array([dN1_dsy[0][5],dN1_dsy[1][5],dN1_dsy[2][5]]).reshape((3,1)))
        dN2_GP6=np.matmul(Jinv, np.array([dN2_dsy[0][5],dN2_dsy[1][5],dN2_dsy[2][5]]).reshape((3,1)))
        dN3_GP6=np.matmul(Jinv, np.array([dN3_dsy[0][5],dN3_dsy[1][5],dN3_dsy[2][5]]).reshape((3,1)))
        dN4_GP6=np.matmul(Jinv, np.array([dN4_dsy[0][5],dN4_dsy[1][5],dN4_dsy[2][5]]).reshape((3,1)))
        dN5_GP6=np.matmul(Jinv, np.array([dN5_dsy[0][5],dN5_dsy[1][5],dN5_dsy[2][5]]).reshape((3,1)))
        dN6_GP6=np.matmul(Jinv, np.array([dN6_dsy[0][5],dN6_dsy[1][5],dN6_dsy[2][5]]).reshape((3,1)))
        dN7_GP6=np.matmul(Jinv, np.array([dN7_dsy[0][5],dN7_dsy[1][5],dN7_dsy[2][5]]).reshape((3,1)))
        dN8_GP6=np.matmul(Jinv, np.array([dN8_dsy[0][5],dN8_dsy[1][5],dN8_dsy[2][5]]).reshape((3,1)))

        dN1_GP7=np.matmul(Jinv, np.array([dN1_dsy[0][6],dN1_dsy[1][6],dN1_dsy[2][6]]).reshape((3,1)))
        dN2_GP7=np.matmul(Jinv, np.array([dN2_dsy[0][6],dN2_dsy[1][6],dN2_dsy[2][6]]).reshape((3,1)))
        dN3_GP7=np.matmul(Jinv, np.array([dN3_dsy[0][6],dN3_dsy[1][6],dN3_dsy[2][6]]).reshape((3,1)))
        dN4_GP7=np.matmul(Jinv, np.array([dN4_dsy[0][6],dN4_dsy[1][6],dN4_dsy[2][6]]).reshape((3,1)))
        dN5_GP7=np.matmul(Jinv, np.array([dN5_dsy[0][6],dN5_dsy[1][6],dN5_dsy[2][6]]).reshape((3,1)))
        dN6_GP7=np.matmul(Jinv, np.array([dN6_dsy[0][6],dN6_dsy[1][6],dN6_dsy[2][6]]).reshape((3,1)))
        dN7_GP7=np.matmul(Jinv, np.array([dN7_dsy[0][6],dN7_dsy[1][6],dN7_dsy[2][6]]).reshape((3,1)))
        dN8_GP7=np.matmul(Jinv, np.array([dN8_dsy[0][6],dN8_dsy[1][6],dN8_dsy[2][6]]).reshape((3,1)))

        dN1_GP8=np.matmul(Jinv, np.array([dN1_dsy[0][7],dN1_dsy[1][7],dN1_dsy[2][7]]).reshape((3,1)))
        dN2_GP8=np.matmul(Jinv, np.array([dN2_dsy[0][7],dN2_dsy[1][7],dN2_dsy[2][7]]).reshape((3,1)))
        dN3_GP8=np.matmul(Jinv, np.array([dN3_dsy[0][7],dN3_dsy[1][7],dN3_dsy[2][7]]).reshape((3,1)))
        dN4_GP8=np.matmul(Jinv, np.array([dN4_dsy[0][7],dN4_dsy[1][7],dN4_dsy[2][7]]).reshape((3,1)))
        dN5_GP8=np.matmul(Jinv, np.array([dN5_dsy[0][7],dN5_dsy[1][7],dN5_dsy[2][7]]).reshape((3,1)))
        dN6_GP8=np.matmul(Jinv, np.array([dN6_dsy[0][7],dN6_dsy[1][7],dN6_dsy[2][7]]).reshape((3,1)))
        dN7_GP8=np.matmul(Jinv, np.array([dN7_dsy[0][7],dN7_dsy[1][7],dN7_dsy[2][7]]).reshape((3,1)))
        dN8_GP8=np.matmul(Jinv, np.array([dN8_dsy[0][7],dN8_dsy[1][7],dN8_dsy[2][7]]).reshape((3,1)))

        dN = [dN1_GP1, dN1_GP2, dN1_GP3, dN1_GP4, dN1_GP5, dN1_GP6, dN1_GP7, dN1_GP8,
              dN2_GP1, dN2_GP2, dN2_GP3, dN2_GP4, dN2_GP5, dN2_GP6, dN2_GP7, dN2_GP8,
              dN3_GP1, dN3_GP2, dN3_GP3, dN3_GP4, dN3_GP5, dN3_GP6, dN3_GP7, dN3_GP8,
              dN4_GP1, dN4_GP2, dN4_GP3, dN4_GP4, dN4_GP5, dN4_GP6, dN4_GP7, dN4_GP8,
              dN5_GP1, dN5_GP2, dN5_GP3, dN5_GP4, dN5_GP5, dN5_GP6, dN5_GP7, dN5_GP8,
              dN6_GP1, dN6_GP2, dN6_GP3, dN6_GP4, dN6_GP5, dN6_GP6, dN6_GP7, dN6_GP8,
              dN7_GP1, dN7_GP2, dN7_GP3, dN7_GP4, dN7_GP5, dN7_GP6, dN7_GP7, dN7_GP8,
              dN8_GP1, dN8_GP2, dN8_GP3, dN8_GP4, dN8_GP5, dN8_GP6, dN8_GP7, dN8_GP8,]
    
    elif Diff_type == 'hex_reduced':
        dN1_dsy=np.array([[-0.125],
                          [-0.125],
                          [-0.125]])
        dN2_dsy=np.array([[0.125],
                          [-0.125],
                          [-0.125]])
        dN3_dsy=np.array([[0.125],
                          [0.125],
                          [-0.125]])
        dN4_dsy=np.array([[-0.125],
                          [0.125],
                          [-0.125]])
        dN5_dsy=np.array([[-0.125],
                          [-0.125],
                          [0.125]])
        dN6_dsy=np.array([[0.125],
                          [-0.125],
                          [0.125]])
        dN7_dsy=np.array([[0.125],
                          [0.125],
                          [0.125]])
        dN8_dsy=np.array([[-0.125],
                          [0.125],
                          [0.125]])

        dN1_GP1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0],dN1_dsy[2][0]]).reshape((3,1)))
        dN2_GP1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0],dN2_dsy[2][0]]).reshape((3,1)))
        dN3_GP1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0],dN3_dsy[2][0]]).reshape((3,1)))
        dN4_GP1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0],dN4_dsy[2][0]]).reshape((3,1)))
        dN5_GP1=np.matmul(Jinv, np.array([dN5_dsy[0][0],dN5_dsy[1][0],dN5_dsy[2][0]]).reshape((3,1)))
        dN6_GP1=np.matmul(Jinv, np.array([dN6_dsy[0][0],dN6_dsy[1][0],dN6_dsy[2][0]]).reshape((3,1)))
        dN7_GP1=np.matmul(Jinv, np.array([dN7_dsy[0][0],dN7_dsy[1][0],dN7_dsy[2][0]]).reshape((3,1)))
        dN8_GP1=np.matmul(Jinv, np.array([dN8_dsy[0][0],dN8_dsy[1][0],dN8_dsy[2][0]]).reshape((3,1)))

        dN = [dN1_GP1, dN2_GP1, dN3_GP1, dN4_GP1, dN5_GP1, dN6_GP1, dN7_GP1, dN8_GP1]

    else:
        dN = None
    return dN

def get_data_EX3D1(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelz = MP['Nelz']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    Nelz_max = MP['Nelz_max']
    Nelz_min = MP['Nelz_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    zmin = MP['domain']['z'][0]
    zmax = MP['domain']['z'][1]
    Diff_type = MP['Diff_type']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    zi = np.linspace(zmin, zmax, num=Nelz+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]
    xi, yi, zi = np.meshgrid(xi, yi, zi)
    xi = xi.transpose(1,0,2)
    yi = yi.transpose(1,0,2)
    zi = zi.transpose(1,0,2)
    J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_3D(Diff_type, Jinv)

   # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col = mask_domain
    Nx = mask_col.shape[0]
    Ny = mask_col.shape[1]
    Nz = mask_col.shape[2]
    mask_col = mask_col.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)

    # create meshgrid for fem element center
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1,0:-1]
    zi_elem = zi + dz/2.0
    zi_elem = zi_elem[0:-1,0:-1,0:-1]
    distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col_elem = mask_domain.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
    # X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    # # debug
    # # transfer mesh to nodal vector: xi.flatten()
    # # transfer nodal vector to mesh: x_node.reshape(Nx,Ny,Nz) 
    # x_node = X_col[:,0]
    # y_node = X_col[:,1]
    # z_node = X_col[:,2]
    # xi_from_node = x_node.reshape(Nx,Ny,Nz)
    # yi_from_node = y_node.reshape(Nx,Ny,Nz)
    # zi_from_node = z_node.reshape(Nx,Ny,Nz)
    # node_ID = torch.arange(1, Nx * Ny * Nz + 1)
    # node_ID_mesh = node_ID.reshape(Nx,Ny,Nz)
    
    # # calculate the element center coordinates:
    # # transfer mesh to element vector: xi_elem.flatten()
    # # transfer element vector to mesh: x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # xi_elem = xi + dx/2.0
    # xi_elem = xi_elem[0:-1,0:-1,0:-1]
    # yi_elem = yi + dy/2.0
    # yi_elem = yi_elem[0:-1,0:-1,0:-1]
    # zi_elem = zi + dz/2.0
    # zi_elem = zi_elem[0:-1,0:-1,0:-1]
    
    # distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    # mask_domain = distance > -1
    # mask_col_elem = mask_domain
    # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)

    # x_elem = X_col_elem[:,0]
    # y_elem = X_col_elem[:,1]
    # z_elem = X_col_elem[:,2]
    # # transfer node to element center
    # xi_from_elem = x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # yi_from_elem = y_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # zi_from_elem = z_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # elem_ID = torch.arange(1, (Nx-1) * (Ny-1) * (Nz-1) + 1)
    # elem_ID_mesh = elem_ID.reshape((Nx-1),(Ny-1),(Nz-1))
    # rho_mesh = elem_ID_mesh
    
    # # calculate element rho:
    # rho_node_mesh = node_ID_mesh
    # rho_elem_mesh = 0.125 * ( rho_node_mesh[:-1, :-1, :-1] + rho_node_mesh[:-1, :-1, 1:] +
    #                           rho_node_mesh[:-1, 1:, :-1] + rho_node_mesh[:-1, 1:, 1:] +
    #                           rho_node_mesh[1:, :-1, :-1] + rho_node_mesh[1:, :-1, 1:] +
    #                           rho_node_mesh[1:, 1:, :-1] + rho_node_mesh[1:, 1:, 1:] )
    # # get correct node numbers for shape functions
    # N1 = node_ID_mesh[:-1,:-1,:-1]
    # N2 = node_ID_mesh[1:,:-1,:-1]
    # N3 = node_ID_mesh[1:,1:,:-1]
    # N4 = node_ID_mesh[:-1,1:,:-1]
    # N5 = node_ID_mesh[:-1,:-1,1:]
    # N6 = node_ID_mesh[1:,:-1,1:]
    # N7 = node_ID_mesh[1:,1:,1:]
    # N8 = node_ID_mesh[:-1,1:,1:]

    LE_index = (X_col[:,0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax) & (X_col[:, 2] >= zmin) & (X_col[:, 2] <= zmax)
    LE = X_col[LE_index] # fixed edge coordinates

    BE_index = (X_col[:,0] == xmax) & (X_col[:,1] == ymin) & (X_col[:, 2] >= zmin) & (X_col[:, 2] <= zmax)
    BE = X_col[BE_index] 

    # define the traction BC:
    line_segment = (zmax - zmin) * 0.1
    traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
    traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
    traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)

    # masked_domain = X_col[mask_col]
    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], X_col[::step,2:3], marker='o', alpha=0.1, s=2, color='green', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2],masked_domain[::step,2:3],marker='o', alpha=0.3, s=2, color='blue', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')
    # ax.scatter(X_col[traction_indices,0], X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s=10, color='black', label = 'external force')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # # Create subplots
    # step = 1
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # # XY Surface View
    # print(X_col[traction_indices,0])
    # axs[0].scatter(X_col[::step, 0], X_col[::step, 1], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[0].scatter(masked_domain[::step, 0], masked_domain[::step, 1], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[0].scatter(X_col[traction_indices,0], X_col[traction_indices,1], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[0].set_xlabel('X (mm)')
    # axs[0].set_ylabel('Y (mm)')
    # axs[0].set_title("XY Surface View")
    # axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[0].axis('equal')

    # # XZ Surface View
    # axs[1].scatter(X_col[::step, 0], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[1].scatter(masked_domain[::step, 0], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[1].scatter(X_col[traction_indices,0], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[1].set_xlabel('X (mm)')
    # axs[1].set_ylabel('Z (mm)')
    # axs[1].set_title("XZ Surface View")
    # axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[1].axis('equal')

    # # YZ Surface View
    # axs[2].scatter(X_col[::step, 1], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[2].scatter(masked_domain[::step, 1], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[2].scatter(X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[2].set_xlabel('Y (mm)')
    # axs[2].set_ylabel('Z (mm)')
    # axs[2].set_title("YZ Surface View")
    # axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[2].axis('equal')

    # plt.tight_layout()
    # plt.show()

    # define the training dataset
    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_coord = torch.tensor(X_col[traction_indices]).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(BE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    w_X_train = torch.tensor(BE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    w_train = (torch.zeros_like(w_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi, zi= np.linspace(xmin, xmax, 2*Nelx), np.linspace(ymin, ymax, 2*Nely), np.linspace(zmin, zmax, 2*Nelz)
    xi, yi, zi= np.meshgrid(xi, yi, zi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2 + (zi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,'w_X_train':w_X_train,'w_train':w_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'dz':dz,'Nx':Nx,'Ny':Ny,'Nz':Nz, 'mask_nan':mask_nan,'xi':xi,'yi':yi,'zi':zi,
                'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    Nelz_list= np.floor(np.linspace(Nelz_min,Nelz_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        Nelz = Nelz_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        dN = get_dN_3D(Diff_type, Jinv)

        # delete points in unwanted area:
        distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col = mask_domain
        Nx = mask_col.shape[0]
        Ny = mask_col.shape[1]
        Nz = mask_col.shape[2]
        mask_col = mask_col.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # create meshgrid for fem element center
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col_elem = mask_domain.flatten()
        X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
        X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
        
        # define the traction BC:
        line_segment = (zmax - zmin) * 0.1
        traction_indices = (X_col[:, 0] == xmin) & (X_col[:, 1] == ymax) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
        traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
        traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)

        # Save all data in a dictionary
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,
                      'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'Nz':Nz,'dx':dx,'dy':dy,'dz':dz,
                      'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX3D2(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelz = MP['Nelz']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    Nelz_max = MP['Nelz_max']
    Nelz_min = MP['Nelz_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    zmin = MP['domain']['z'][0]
    zmax = MP['domain']['z'][1]
    Diff_type = MP['Diff_type']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    zi = np.linspace(zmin, zmax, num=Nelz+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]
    xi, yi, zi = np.meshgrid(xi, yi, zi)
    xi = xi.transpose(1,0,2)
    yi = yi.transpose(1,0,2)
    zi = zi.transpose(1,0,2)
    J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_3D(Diff_type, Jinv)

   # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col = mask_domain
    Nx = mask_col.shape[0]
    Ny = mask_col.shape[1]
    Nz = mask_col.shape[2]
    mask_col = mask_col.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)

    # create meshgrid for fem element center
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1,0:-1]
    zi_elem = zi + dz/2.0
    zi_elem = zi_elem[0:-1,0:-1,0:-1]
    distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col_elem = mask_domain.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
    X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    # # debug
    # # transfer mesh to nodal vector: xi.flatten()
    # # transfer nodal vector to mesh: x_node.reshape(Nx,Ny,Nz) 
    # x_node = X_col[:,0]
    # y_node = X_col[:,1]
    # z_node = X_col[:,2]
    # xi_from_node = x_node.reshape(Nx,Ny,Nz)
    # yi_from_node = y_node.reshape(Nx,Ny,Nz)
    # zi_from_node = z_node.reshape(Nx,Ny,Nz)
    # node_ID = torch.arange(1, Nx * Ny * Nz + 1)
    # node_ID_mesh = node_ID.reshape(Nx,Ny,Nz)
    
    # # calculate the element center coordinates:
    # # transfer mesh to element vector: xi_elem.flatten()
    # # transfer element vector to mesh: x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # xi_elem = xi + dx/2.0
    # xi_elem = xi_elem[0:-1,0:-1,0:-1]
    # yi_elem = yi + dy/2.0
    # yi_elem = yi_elem[0:-1,0:-1,0:-1]
    # zi_elem = zi + dz/2.0
    # zi_elem = zi_elem[0:-1,0:-1,0:-1]
    
    # distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    # mask_domain = distance > -1
    # mask_col_elem = mask_domain
    # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)

    # x_elem = X_col_elem[:,0]
    # y_elem = X_col_elem[:,1]
    # z_elem = X_col_elem[:,2]
    # # transfer node to element center
    # xi_from_elem = x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # yi_from_elem = y_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # zi_from_elem = z_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # elem_ID = torch.arange(1, (Nx-1) * (Ny-1) * (Nz-1) + 1)
    # elem_ID_mesh = elem_ID.reshape((Nx-1),(Ny-1),(Nz-1))
    # rho_mesh = elem_ID_mesh
    
    # # calculate element rho:
    # rho_node_mesh = node_ID_mesh
    # rho_elem_mesh = 0.125 * ( rho_node_mesh[:-1, :-1, :-1] + rho_node_mesh[:-1, :-1, 1:] +
    #                           rho_node_mesh[:-1, 1:, :-1] + rho_node_mesh[:-1, 1:, 1:] +
    #                           rho_node_mesh[1:, :-1, :-1] + rho_node_mesh[1:, :-1, 1:] +
    #                           rho_node_mesh[1:, 1:, :-1] + rho_node_mesh[1:, 1:, 1:] )
    # # get correct node numbers for shape functions
    # N1 = node_ID_mesh[:-1,:-1,:-1]
    # N2 = node_ID_mesh[1:,:-1,:-1]
    # N3 = node_ID_mesh[1:,1:,:-1]
    # N4 = node_ID_mesh[:-1,1:,:-1]
    # N5 = node_ID_mesh[:-1,:-1,1:]
    # N6 = node_ID_mesh[1:,:-1,1:]
    # N7 = node_ID_mesh[1:,1:,1:]
    # N8 = node_ID_mesh[:-1,1:,1:]

    FE_index = (X_col[:,0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax) & (X_col[:, 2] >= zmin) & (X_col[:, 2] <= zmax)
    FE = X_col[FE_index] # fixed edge coordinates

    # define the traction BC:
    line_segment = (zmax - zmin) * 0.1
    traction_indices = (X_col[:, 0] == xmax) & (X_col[:, 1] == ymin) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
    traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
    traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)

    # masked_domain = X_col[mask_col]
    # # Visualize ALL the training points
    # step = 2
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], X_col[::step,2:3], marker='o', alpha=0.1, s=2, color='green', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2],masked_domain[::step,2:3],marker='o', alpha=0.3, s=2, color='blue', label = 'masked domain')
    # ax.scatter(FE[::1,0:1], FE[::1,1:2], FE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'fixed edge')
    # ax.scatter(X_col[traction_indices,0], X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s=10, color='black', label = 'external force')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # # Create subplots
    # step = 1
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # # XY Surface View
    # print(X_col[traction_indices,0])
    # axs[0].scatter(X_col[::step, 0], X_col[::step, 1], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[0].scatter(masked_domain[::step, 0], masked_domain[::step, 1], alpha=0.3, s=2, color='blue', label='Masked domain')
    # axs[0].scatter(FE[:, 0], FE[:, 1], alpha=0.9, s=2, color='red', label='Fixed edge')
    # axs[0].scatter(X_col[traction_indices,0], X_col[traction_indices,1], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[0].set_xlabel('X (mm)')
    # axs[0].set_ylabel('Y (mm)')
    # axs[0].set_title("XY Surface View")
    # axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[0].axis('equal')

    # # XZ Surface View
    # axs[1].scatter(X_col[::step, 0], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[1].scatter(masked_domain[::step, 0], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # axs[1].scatter(FE[:, 0], FE[:, 2], alpha=0.9, s=2, color='red', label='Fixed edge')
    # axs[1].scatter(X_col[traction_indices,0], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[1].set_xlabel('X (mm)')
    # axs[1].set_ylabel('Z (mm)')
    # axs[1].set_title("XZ Surface View")
    # axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[1].axis('equal')

    # # YZ Surface View
    # axs[2].scatter(X_col[::step, 1], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[2].scatter(masked_domain[::step, 1], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # axs[2].scatter(FE[:, 1], FE[:, 2], alpha=0.9, s=2, color='red', label='Fixed edge')
    # axs[2].scatter(X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[2].set_xlabel('Y (mm)')
    # axs[2].set_ylabel('Z (mm)')
    # axs[2].set_title("YZ Surface View")
    # axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[2].axis('equal')

    # plt.tight_layout()
    # plt.show()

    # define the training dataset
    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    FE = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_coord = torch.tensor(X_col[traction_indices]).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

    u_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    w_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    w_train = (torch.zeros_like(w_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi, zi= np.linspace(xmin, xmax, 2*Nelx), np.linspace(ymin, ymax, 2*Nely), np.linspace(zmin, zmax, 2*Nelz)
    xi, yi, zi= np.meshgrid(xi, yi, zi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2 + (zi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,'w_X_train':w_X_train,'w_train':w_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'dz':dz,'Nx':Nx,'Ny':Ny,'Nz':Nz, 'mask_nan':mask_nan,'xi':xi,'yi':yi,'zi':zi,
                'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    Nelz_list= np.floor(np.linspace(Nelz_min,Nelz_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        Nelz = Nelz_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        dN = get_dN_3D(Diff_type, Jinv)

        # delete points in unwanted area:
        distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col = mask_domain
        Nx = mask_col.shape[0]
        Ny = mask_col.shape[1]
        Nz = mask_col.shape[2]
        mask_col = mask_col.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # create meshgrid for fem element center
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col_elem = mask_domain.flatten()
        X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
        X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
        
        # define the traction BC:
        line_segment = (zmax - zmin) * 0.1
        traction_indices = (X_col[:, 0] == xmax) & (X_col[:, 1] == ymin) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
        traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
        traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)

        # Save all data in a dictionary
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,
                      'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'Nz':Nz,'dx':dx,'dy':dy,'dz':dz,
                      'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX3D3(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelz = MP['Nelz']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    Nelz_max = MP['Nelz_max']
    Nelz_min = MP['Nelz_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    zmin = MP['domain']['z'][0]
    zmax = MP['domain']['z'][1]
    Diff_type = MP['Diff_type']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    zi = np.linspace(zmin, zmax, num=Nelz+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]
    xi, yi, zi = np.meshgrid(xi, yi, zi)
    xi = xi.transpose(1,0,2)
    yi = yi.transpose(1,0,2)
    zi = zi.transpose(1,0,2)
    J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_3D(Diff_type, Jinv)

   # delete points in unwanted area:
    distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col = mask_domain
    Nx = mask_col.shape[0]
    Ny = mask_col.shape[1]
    Nz = mask_col.shape[2]
    mask_col = mask_col.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)

    # create meshgrid for fem element center
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1,0:-1]
    zi_elem = zi + dz/2.0
    zi_elem = zi_elem[0:-1,0:-1,0:-1]
    distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    mask_domain = distance > -1
    mask_col_elem = mask_domain.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
    # X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

    # # debug
    # # transfer mesh to nodal vector: xi.flatten()
    # # transfer nodal vector to mesh: x_node.reshape(Nx,Ny,Nz) 
    # x_node = X_col[:,0]
    # y_node = X_col[:,1]
    # z_node = X_col[:,2]
    # xi_from_node = x_node.reshape(Nx,Ny,Nz)
    # yi_from_node = y_node.reshape(Nx,Ny,Nz)
    # zi_from_node = z_node.reshape(Nx,Ny,Nz)
    # node_ID = torch.arange(1, Nx * Ny * Nz + 1)
    # node_ID_mesh = node_ID.reshape(Nx,Ny,Nz)
    
    # # calculate the element center coordinates:
    # # transfer mesh to element vector: xi_elem.flatten()
    # # transfer element vector to mesh: x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # xi_elem = xi + dx/2.0
    # xi_elem = xi_elem[0:-1,0:-1,0:-1]
    # yi_elem = yi + dy/2.0
    # yi_elem = yi_elem[0:-1,0:-1,0:-1]
    # zi_elem = zi + dz/2.0
    # zi_elem = zi_elem[0:-1,0:-1,0:-1]
    
    # distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
    # mask_domain = distance > -1
    # mask_col_elem = mask_domain
    # X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)

    # x_elem = X_col_elem[:,0]
    # y_elem = X_col_elem[:,1]
    # z_elem = X_col_elem[:,2]
    # # transfer node to element center
    # xi_from_elem = x_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # yi_from_elem = y_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # zi_from_elem = z_elem.reshape((Nx-1),(Ny-1),(Nz-1))
    # elem_ID = torch.arange(1, (Nx-1) * (Ny-1) * (Nz-1) + 1)
    # elem_ID_mesh = elem_ID.reshape((Nx-1),(Ny-1),(Nz-1))
    # rho_mesh = elem_ID_mesh
    
    # # calculate element rho:
    # rho_node_mesh = node_ID_mesh
    # rho_elem_mesh = 0.125 * ( rho_node_mesh[:-1, :-1, :-1] + rho_node_mesh[:-1, :-1, 1:] +
    #                           rho_node_mesh[:-1, 1:, :-1] + rho_node_mesh[:-1, 1:, 1:] +
    #                           rho_node_mesh[1:, :-1, :-1] + rho_node_mesh[1:, :-1, 1:] +
    #                           rho_node_mesh[1:, 1:, :-1] + rho_node_mesh[1:, 1:, 1:] )
    # # get correct node numbers for shape functions
    # N1 = node_ID_mesh[:-1,:-1,:-1]
    # N2 = node_ID_mesh[1:,:-1,:-1]
    # N3 = node_ID_mesh[1:,1:,:-1]
    # N4 = node_ID_mesh[:-1,1:,:-1]
    # N5 = node_ID_mesh[:-1,:-1,1:]
    # N6 = node_ID_mesh[1:,:-1,1:]
    # N7 = node_ID_mesh[1:,1:,1:]
    # N8 = node_ID_mesh[:-1,1:,1:]

    LE_index = (X_col[:,0] == 0.0) & (X_col[:, 1] >= ymin) & (X_col[:, 1] <= ymax) & (X_col[:, 2] >= zmin) & (X_col[:, 2] <= zmax)
    LE = X_col[LE_index] # fixed edge coordinates

    BE_index = (X_col[:,0] == xmax) & (X_col[:,1] == ymin) & (X_col[:, 2] >= zmin) & (X_col[:, 2] <= zmax)
    BE = X_col[BE_index] 

    # define the traction BC:
    line_segment = (zmax - zmin) * 0.1
    center_z = (zmax + zmin)/2
    center_x = (xmax + xmin)/2
    traction_indices1 = (X_col[:, 1] == ymin) & (X_col[:,0] >= 0.0) & (X_col[:,0] <= line_segment/2) & (X_col[:,2] >= (center_z - line_segment/2)) & (X_col[:,2] <= (center_z + line_segment/2))
    traction_indices1 = torch.nonzero(traction_indices1,as_tuple=False).view(-1)
    traction_magnitude1 = torch.tensor(-0.1)/len(traction_indices1) * torch.ones_like(traction_indices1)
    traction_indices2 = (X_col[:, 1] == ymin) & (X_col[:,0] >= (center_x - line_segment/2)) & (X_col[:,0] <= (center_x + line_segment/2)) & (X_col[:,2] >= (center_z - line_segment/2)) & (X_col[:,2] <= (center_z + line_segment/2))
    traction_indices2 = torch.nonzero(traction_indices2,as_tuple=False).view(-1)
    traction_magnitude2 = torch.tensor(-0.1)/len(traction_indices2) * torch.ones_like(traction_indices2)
    traction_indices = torch.cat([traction_indices1, traction_indices2], dim=0)
    traction_magnitude = torch.cat([traction_magnitude1, traction_magnitude2], dim=0)
    
    # masked_domain = X_col[mask_col]
    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], X_col[::step,2:3], marker='o', alpha=0.1, s=2, color='green', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2],masked_domain[::step,2:3],marker='o', alpha=0.3, s=2, color='blue', label = 'masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')
    # ax.scatter(X_col[traction_indices,0], X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s=10, color='black', label = 'external force')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()

    # # Create subplots
    # step = 1
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # # XY Surface View
    # print(X_col[traction_indices,0])
    # axs[0].scatter(X_col[::step, 0], X_col[::step, 1], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[0].scatter(masked_domain[::step, 0], masked_domain[::step, 1], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[0].scatter(X_col[traction_indices,0], X_col[traction_indices,1], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[0].set_xlabel('X (mm)')
    # axs[0].set_ylabel('Y (mm)')
    # axs[0].set_title("XY Surface View")
    # axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[0].axis('equal')

    # # XZ Surface View
    # axs[1].scatter(X_col[::step, 0], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[1].scatter(masked_domain[::step, 0], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[1].scatter(X_col[traction_indices,0], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[1].set_xlabel('X (mm)')
    # axs[1].set_ylabel('Z (mm)')
    # axs[1].set_title("XZ Surface View")
    # axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[1].axis('equal')

    # # YZ Surface View
    # axs[2].scatter(X_col[::step, 1], X_col[::step, 2], alpha=0.1, s=2, color='green', label='All grid points')
    # axs[2].scatter(masked_domain[::step, 1], masked_domain[::step, 2], alpha=0.3, s=2, color='blue', label='Masked domain')
    # ax.scatter(LE[::1,0:1], LE[::1,1:2], LE[::1,2:3], marker='o', alpha=0.9, s=2, color='orange', label = 'left surface')
    # ax.scatter(BE[::1,0:1], BE[::1,1:2], BE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'bottom edge')    
    # axs[2].scatter(X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s = 5, color='black', label = 'external force')
    # axs[2].set_xlabel('Y (mm)')
    # axs[2].set_ylabel('Z (mm)')
    # axs[2].set_title("YZ Surface View")
    # axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1))
    # axs[2].axis('equal')

    # plt.tight_layout()
    # plt.show()

    # define the training dataset
    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    LE = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_coord = torch.tensor(X_col[traction_indices]).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

    u_X_train = torch.tensor(LE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(BE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    w_X_train = torch.tensor(BE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    w_train = (torch.zeros_like(w_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    rho_X_train = traction_coord.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(rho_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi, zi= np.linspace(xmin, xmax, 2*Nelx), np.linspace(ymin, ymax, 2*Nely), np.linspace(zmin, zmax, 2*Nelz)
    xi, yi, zi= np.meshgrid(xi, yi, zi)
    distance = np.sqrt((xi) ** 2 + (yi) ** 2 + (zi) ** 2)
    mask_nan = distance < -1 # no hole in this example, mask_nan is all false

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,'w_X_train':w_X_train,'w_train':w_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'dz':dz,'Nx':Nx,'Ny':Ny,'Nz':Nz, 'mask_nan':mask_nan,'xi':xi,'yi':yi,'zi':zi,
                'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    Nelz_list= np.floor(np.linspace(Nelz_min,Nelz_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        Nelz = Nelz_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        dN = get_dN_3D(Diff_type, Jinv)

        # delete points in unwanted area:
        distance = ((xi) ** 2 + (yi) ** 2 + (zi) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col = mask_domain
        Nx = mask_col.shape[0]
        Ny = mask_col.shape[1]
        Nz = mask_col.shape[2]
        mask_col = mask_col.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # create meshgrid for fem element center
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        distance = ((xi_elem) ** 2 + (yi_elem) ** 2 + (zi_elem) ** 2) ** 0.5
        mask_domain = distance > -1
        mask_col_elem = mask_domain.flatten()
        X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
        X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
        
        # define the traction BC:
        line_segment = (zmax - zmin) * 0.1
        center_z = (zmax + zmin)/2
        center_x = (xmax + xmin)/2
        traction_indices1 = (X_col[:, 1] == ymin) & (X_col[:,0] >= 0.0) & (X_col[:,0] <= line_segment/2) & (X_col[:,2] >= (center_z - line_segment/2)) & (X_col[:,2] <= (center_z + line_segment/2))
        traction_indices1 = torch.nonzero(traction_indices1,as_tuple=False).view(-1)
        traction_magnitude1 = torch.tensor(-0.1)/len(traction_indices1) * torch.ones_like(traction_indices1)
        traction_indices2 = (X_col[:, 1] == ymin) & (X_col[:,0] >= (center_x - line_segment/2)) & (X_col[:,0] <= (center_x + line_segment/2)) & (X_col[:,2] >= (center_z - line_segment/2)) & (X_col[:,2] <= (center_z + line_segment/2))
        traction_indices2 = torch.nonzero(traction_indices2,as_tuple=False).view(-1)
        traction_magnitude2 = torch.tensor(-0.1)/len(traction_indices2) * torch.ones_like(traction_indices2)
        traction_indices = torch.cat([traction_indices1, traction_indices2], dim=0)
        traction_magnitude = torch.cat([traction_magnitude1, traction_magnitude2], dim=0)
    
        # Save all data in a dictionary
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,
                      'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'Nz':Nz,'dx':dx,'dy':dy,'dz':dz,
                      'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

def get_data_EX3D4(MP):
    # get collocation points
    Nelx = MP['Nelx']
    Nely = MP['Nely']
    Nelz = MP['Nelz']
    Nelx_max = MP['Nelx_max']
    Nelx_min = MP['Nelx_min']
    Nely_max = MP['Nely_max']
    Nely_min = MP['Nely_min']
    Nelz_max = MP['Nelz_max']
    Nelz_min = MP['Nelz_min']
    num_CP = MP['num_CP']
    xmin = MP['domain']['x'][0]
    xmax = MP['domain']['x'][1]
    ymin = MP['domain']['y'][0]
    ymax = MP['domain']['y'][1]
    zmin = MP['domain']['z'][0]
    zmax = MP['domain']['z'][1]
    hole_center = MP['hole_center']
    hole_radius = MP['hole_radius']
    ring_thickness = MP['ring_thickness']
    Diff_type = MP['Diff_type']
    xi = np.linspace(xmin, xmax, num=Nelx+1)
    yi = np.linspace(ymin, ymax, num=Nely+1)
    zi = np.linspace(zmin, zmax, num=Nelz+1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]
    xi, yi, zi = np.meshgrid(xi, yi, zi)
    xi = xi.transpose(1,0,2)
    yi = yi.transpose(1,0,2)
    zi = zi.transpose(1,0,2)
    J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
    Jinv= np.linalg.inv(J)
    detJ= np.linalg.det(J)
    dN = get_dN_3D(Diff_type, Jinv)

   # delete points in unwanted area:
    mask_domain = (xi <= 20.0) | (yi <= 20.0) 
    mask_col = mask_domain

    Nx = mask_col.shape[0]
    Ny = mask_col.shape[1]
    Nz = mask_col.shape[2]
    mask_col = mask_col.flatten() # reshape it to the same size with X_col
    X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)

    # create meshgrid for fem element center
    xi_elem = xi + dx/2.0
    xi_elem = xi_elem[0:-1,0:-1,0:-1]
    yi_elem = yi + dy/2.0
    yi_elem = yi_elem[0:-1,0:-1,0:-1]
    zi_elem = zi + dz/2.0
    zi_elem = zi_elem[0:-1,0:-1,0:-1]
    mask_domain = (xi_elem <= 20.0) | (yi_elem <= 20.0) 

    mask_col_elem = mask_domain.flatten()
    X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)

    FE_index = (X_col[:,0] >= 0.0) & (X_col[:,0] <= 20.0) & (X_col[:, 1] == ymax)
    FE = X_col[FE_index] # fixed edge coordinates

    # define the traction BC:
    line_segment = (zmax - zmin) * 0.1
    traction_coord_xy = torch.tensor([[xmax, 20.0]])
    masked_domain = X_col[mask_col]
    distances = torch.norm(masked_domain[:,0:2] - traction_coord_xy, dim=1)
    min_dist = distances.min()
    closest_point_xy_in_mask = (distances == min_dist).nonzero(as_tuple=True)[0]    
    Coord_masked = masked_domain[closest_point_xy_in_mask]
    traction_indices = (X_col[:, 0] == Coord_masked[0,0]) & (X_col[:, 1] == Coord_masked[0,1]) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
    traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
    traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)
    
    # masked_domain = X_col[mask_col]
    # # Visualize ALL the training points
    # step = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(X_col[::step,0:1], X_col[::step,1:2], X_col[::step,2:3], marker='o', alpha=0.1, s=2, color='green', label = 'All grid points')
    # ax.scatter(masked_domain[::step,0:1], masked_domain[::step,1:2],masked_domain[::step,2:3],marker='o', alpha=0.3, s=2, color='blue', label = 'masked domain')
    # ax.scatter(FE[::1,0:1], FE[::1,1:2], FE[::1,2:3], marker='o', alpha=0.9, s=2, color='red', label = 'fixed edge')
    # ax.scatter(X_col[traction_indices,0], X_col[traction_indices,1], X_col[traction_indices,2], marker='o', alpha=0.9, s=10, color='black', label = 'external force')

    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_aspect('equal')
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.show()


    # define the training dataset
    X_col = torch.tensor(X_col).type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
    traction_magnitude = traction_magnitude.type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    FE = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    traction_coord = torch.tensor(X_col[traction_indices]).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

    u_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    u_train = (torch.zeros_like(u_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    v_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    v_train = (torch.zeros_like(v_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    w_X_train = torch.tensor(FE).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    w_train = (torch.zeros_like(w_X_train)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])
    
    rho_X_train = torch.tensor(traction_coord).type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
    rho_train = (torch.ones_like(traction_coord)[:,0]).type(tkwargs["dtype"]).to(tkwargs['device'])

    # define the mask and meshgrid for plotting only (should not include padding layers)
    xi, yi, zi= np.linspace(xmin, xmax, 2*Nelx), np.linspace(ymin, ymax, 2*Nely), np.linspace(zmin, zmax, 2*Nelz)
    xi, yi, zi= np.meshgrid(xi, yi, zi)
    # distance = np.sqrt((xi) ** 2 + (yi) ** 2 + (zi) ** 2)
    mask_nan = (xi >= 20.0) & (yi>=20.0)

    Training = {'u_X_train':u_X_train,'u_train':u_train,'v_X_train':v_X_train,'v_train':v_train,'w_X_train':w_X_train,'w_train':w_train,
                'rho_X_train':rho_X_train,'rho_train':rho_train,'traction_magnitude':traction_magnitude,'traction_indices':traction_indices,
                'X_col':X_col,'mask_col':mask_col,'dx':dx, 'dy':dy,'dz':dz,'Nx':Nx,'Ny':Ny,'Nz':Nz, 'mask_nan':mask_nan,'xi':xi,'yi':yi,'zi':zi,
                'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
    
    # define all other collocation points
    Nelx_list= np.floor(np.linspace(Nelx_min,Nelx_max,num_CP)).astype(int)
    Nely_list= np.floor(np.linspace(Nely_min,Nely_max,num_CP)).astype(int)
    Nelz_list= np.floor(np.linspace(Nelz_min,Nelz_max,num_CP)).astype(int)
    X_col_all = []  # Initialize an empty list to store all results
    for i in range(len(Nelx_list)):
        Nelx = Nelx_list[i]
        Nely = Nely_list[i]
        Nelz = Nelz_list[i]
        xi = np.linspace(xmin, xmax, num=Nelx+1)
        yi = np.linspace(ymin, ymax, num=Nely+1)
        zi = np.linspace(zmin, zmax, num=Nelz+1)
        dx = xi[1] - xi[0]
        dy = yi[1] - yi[0]
        dz = zi[1] - zi[0]
        xi, yi, zi = np.meshgrid(xi, yi, zi)
        xi = xi.transpose(1,0,2)
        yi = yi.transpose(1,0,2)
        zi = zi.transpose(1,0,2)
        J= np.array([[dx/2,0,0],[0,dy/2,0],[0,0,dz/2]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        dN = get_dN_3D(Diff_type, Jinv)

        # delete points in unwanted area:
        mask_domain = (xi <= 20.0) | (yi <= 20.0) 
        mask_col = mask_domain
        Nx = mask_col.shape[0]
        Ny = mask_col.shape[1]
        Nz = mask_col.shape[2]
        mask_col = mask_col.flatten() # reshape it to the same size with X_col
        X_col = torch.tensor(np.vstack([xi.flatten(),yi.flatten(),zi.flatten()]).T)
        X_col = X_col.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])

        # create meshgrid for fem element center
        xi_elem = xi + dx/2.0
        xi_elem = xi_elem[0:-1,0:-1,0:-1]
        yi_elem = yi + dy/2.0
        yi_elem = yi_elem[0:-1,0:-1,0:-1]
        zi_elem = zi + dz/2.0
        zi_elem = zi_elem[0:-1,0:-1,0:-1]
        mask_domain = (xi_elem <= 20.0) | (yi_elem <= 20.0) 
        mask_col_elem = mask_domain.flatten()
        X_col_elem = torch.tensor(np.vstack([xi_elem.flatten(),yi_elem.flatten(),zi_elem.flatten()]).T)
        X_col_elem = X_col_elem.type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
        
        # define the traction BC:
        line_segment = (zmax - zmin) * 0.1
        traction_coord_xy = torch.tensor([[xmax, 20.0]]).to(tkwargs['device'])
        masked_domain = X_col[mask_col]
        distances = torch.norm(masked_domain[:,0:2] - traction_coord_xy, dim=1)
        min_dist = distances.min()
        closest_point_xy_in_mask = (distances == min_dist).nonzero(as_tuple=True)[0]    
        Coord_masked = masked_domain[closest_point_xy_in_mask]
        traction_indices = (X_col[:, 0] == Coord_masked[0,0]) & (X_col[:, 1] == Coord_masked[0,1]) & (X_col[:,2] >= ((zmax + zmin)/2 - line_segment/2)) & (X_col[:,2] <= ((zmax + zmin)/2 + line_segment/2))
        traction_indices = torch.nonzero(traction_indices,as_tuple=False).view(-1)
        traction_magnitude = torch.tensor(-0.2)/len(traction_indices) * torch.ones_like(traction_indices)
    
        # Save all data in a dictionary
        X_col_data = {'traction_magnitude':traction_magnitude,'traction_indices': traction_indices,
                      'mask_col': mask_col,'X_col': X_col,'Nx': Nx,'Ny': Ny,'Nz':Nz,'dx':dx,'dy':dy,'dz':dz,
                      'detJ':detJ,'dN':dN,'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem}
        
        # Append the dictionary to the list
        X_col_all.append(X_col_data)
    return Training, X_col_all

