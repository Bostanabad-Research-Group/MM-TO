
import torch
import numpy as np

class InternalEnergy:
    def __init__(self, Emax, Emin, nu, penal):
        self.Emax = Emax
        self.Emin = Emin
        self.nu = nu
        self.p = penal

    def Elastic2DGaussQuad (self, u, x, dxdydz, shape, density ):
              
        Ux= torch.transpose(u[:, 0].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        Uy= torch.transpose(u[:, 1].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        
        axis=-1
        
        nd = Ux.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        
        UxN1= Ux[:(shape[1]-1)][tuple(slice2)]
        UxN2= Ux[1:shape[1]][tuple(slice2)]
        UxN3= Ux[0:(shape[1]-1)][tuple(slice1)]
        UxN4= Ux[1:shape[1]][tuple(slice1)]
        
        UyN1= Uy[:(shape[1]-1)][tuple(slice2)]
        UyN2= Uy[1:shape[1]][tuple(slice2)]
        UyN3= Uy[0:(shape[1]-1)][tuple(slice1)]
        UyN4= Uy[1:shape[1]][tuple(slice1)]

        # SIMP
        density_SIMP = torch.pow( density , self.p )


        ## Differentiation of shape functions at gauss quadrature points       
        dN1_dsy=np.array([[-0.394337567,-0.105662433,-0.105662433,-0.394337567],[-0.394337567,-0.394337567,-0.105662433,-0.105662433]])
        dN2_dsy=np.array([[-0.105662433,-0.394337567,-0.394337567,-0.105662433],[0.394337567,0.394337567,0.105662433,0.105662433]])
        dN3_dsy=np.array([[0.394337567,0.105662433,0.105662433,0.394337567],[-0.105662433,-0.105662433,-0.394337567,-0.394337567]])
        dN4_dsy=np.array([[0.105662433,0.394337567,0.394337567,0.105662433],[0.105662433,0.105662433,0.394337567,0.394337567]])        
       
        dx=dxdydz[0]
        dy=dxdydz[1]
        dxds=dx/2
        dydt=dy/2
        
        J= np.array([[dxds,0],[0,dydt]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)        
        
        ## Strain energy at GP1
        dN1_dxy_Gp1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0]]).reshape((2,1)))
        dN2_dxy_Gp1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0]]).reshape((2,1)))
        dN3_dxy_Gp1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0]]).reshape((2,1)))
        dN4_dxy_Gp1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0]]).reshape((2,1)))
           
        dUxdx_GP1= dN1_dxy_Gp1[0][0]*UxN1+ dN2_dxy_Gp1[0][0]*UxN2+ dN3_dxy_Gp1[0][0]*UxN3+ dN4_dxy_Gp1[0][0]*UxN4
        dUxdy_GP1= dN1_dxy_Gp1[1][0]*UxN1+ dN2_dxy_Gp1[1][0]*UxN2+ dN3_dxy_Gp1[1][0]*UxN3+ dN4_dxy_Gp1[1][0]*UxN4
           
        dUydx_GP1= dN1_dxy_Gp1[0][0]*UyN1+ dN2_dxy_Gp1[0][0]*UyN2+ dN3_dxy_Gp1[0][0]*UyN3+ dN4_dxy_Gp1[0][0]*UyN4
        dUydy_GP1= dN1_dxy_Gp1[1][0]*UyN1+ dN2_dxy_Gp1[1][0]*UyN2+ dN3_dxy_Gp1[1][0]*UyN3+ dN4_dxy_Gp1[1][0]*UyN4
                   
        
        #Strains at all gauss quadrature points
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_xy_GP1= 0.5*(dUydx_GP1+dUxdy_GP1)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP1= (self.E*(e_xx_GP1+ self.nu*e_yy_GP1)/(1-self.nu**2))*density_SIMP        
        S_yy_GP1= (self.E*(e_yy_GP1+ self.nu*e_xx_GP1)/(1-self.nu**2))*density_SIMP
        S_xy_GP1= (self.E*e_xy_GP1/(1+self.nu))*density_SIMP
        
        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1+ e_yy_GP1*S_yy_GP1+ 2*e_xy_GP1*S_xy_GP1)
        
        ## Strain energy at GP2
        dN1_dxy_Gp2=np.matmul(Jinv, np.array([dN1_dsy[0][1],dN1_dsy[1][1]]).reshape((2,1)))
        dN2_dxy_Gp2=np.matmul(Jinv, np.array([dN2_dsy[0][1],dN2_dsy[1][1]]).reshape((2,1)))
        dN3_dxy_Gp2=np.matmul(Jinv, np.array([dN3_dsy[0][1],dN3_dsy[1][1]]).reshape((2,1)))
        dN4_dxy_Gp2=np.matmul(Jinv, np.array([dN4_dsy[0][1],dN4_dsy[1][1]]).reshape((2,1)))
        
        dUxdx_GP2= dN1_dxy_Gp2[0][0]*UxN1+ dN2_dxy_Gp2[0][0]*UxN2+ dN3_dxy_Gp2[0][0]*UxN3+ dN4_dxy_Gp2[0][0]*UxN4
        dUxdy_GP2= dN1_dxy_Gp2[1][0]*UxN1+ dN2_dxy_Gp2[1][0]*UxN2+ dN3_dxy_Gp2[1][0]*UxN3+ dN4_dxy_Gp2[1][0]*UxN4
        
        dUydx_GP2= dN1_dxy_Gp2[0][0]*UyN1+ dN2_dxy_Gp2[0][0]*UyN2+ dN3_dxy_Gp2[0][0]*UyN3+ dN4_dxy_Gp2[0][0]*UyN4
        dUydy_GP2= dN1_dxy_Gp2[1][0]*UyN1+ dN2_dxy_Gp2[1][0]*UyN2+ dN3_dxy_Gp2[1][0]*UyN3+ dN4_dxy_Gp2[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP2= dUxdx_GP2
        e_yy_GP2= dUydy_GP2
        e_xy_GP2= 0.5*(dUydx_GP2+dUxdy_GP2)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP2= (self.E*(e_xx_GP2+ self.nu*e_yy_GP2)/(1-self.nu**2))*density_SIMP        
        S_yy_GP2= (self.E*(e_yy_GP2+ self.nu*e_xx_GP2)/(1-self.nu**2))*density_SIMP
        S_xy_GP2= (self.E*e_xy_GP2/(1+self.nu))*density_SIMP
        
        strainEnergy_GP2= (e_xx_GP2*S_xx_GP2+ e_yy_GP2*S_yy_GP2+ 2*e_xy_GP2*S_xy_GP2)
        
        ## Strain energy at GP3
        dN1_dxy_GP3=np.matmul(Jinv, np.array([dN1_dsy[0][2],dN1_dsy[1][2]]).reshape((2,1)))
        dN2_dxy_GP3=np.matmul(Jinv, np.array([dN2_dsy[0][2],dN2_dsy[1][2]]).reshape((2,1)))
        dN3_dxy_GP3=np.matmul(Jinv, np.array([dN3_dsy[0][2],dN3_dsy[1][2]]).reshape((2,1)))
        dN4_dxy_GP3=np.matmul(Jinv, np.array([dN4_dsy[0][2],dN4_dsy[1][2]]).reshape((2,1)))
        
        dUxdx_GP3= dN1_dxy_GP3[0][0]*UxN1+ dN2_dxy_GP3[0][0]*UxN2+ dN3_dxy_GP3[0][0]*UxN3+ dN4_dxy_GP3[0][0]*UxN4
        dUxdy_GP3= dN1_dxy_GP3[1][0]*UxN1+ dN2_dxy_GP3[1][0]*UxN2+ dN3_dxy_GP3[1][0]*UxN3+ dN4_dxy_GP3[1][0]*UxN4
        
        dUydx_GP3= dN1_dxy_GP3[0][0]*UyN1+ dN2_dxy_GP3[0][0]*UyN2+ dN3_dxy_GP3[0][0]*UyN3+ dN4_dxy_GP3[0][0]*UyN4
        dUydy_GP3= dN1_dxy_GP3[1][0]*UyN1+ dN2_dxy_GP3[1][0]*UyN2+ dN3_dxy_GP3[1][0]*UyN3+ dN4_dxy_GP3[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP3= dUxdx_GP3
        e_yy_GP3= dUydy_GP3
        e_xy_GP3= 0.5*(dUydx_GP3+dUxdy_GP3)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP3= (self.E*(e_xx_GP3+ self.nu*e_yy_GP3)/(1-self.nu**2))*density_SIMP        
        S_yy_GP3= (self.E*(e_yy_GP3+ self.nu*e_xx_GP3)/(1-self.nu**2))*density_SIMP
        S_xy_GP3= (self.E*e_xy_GP3/(1+self.nu))*density_SIMP
        
        strainEnergy_GP3= (e_xx_GP3*S_xx_GP3+ e_yy_GP3*S_yy_GP3+ 2*e_xy_GP3*S_xy_GP3)
        
        ## Strain energy at GP4
        dN1_dxy_GP4=np.matmul(Jinv, np.array([dN1_dsy[0][3],dN1_dsy[1][3]]).reshape((2,1)))
        dN2_dxy_GP4=np.matmul(Jinv, np.array([dN2_dsy[0][3],dN2_dsy[1][3]]).reshape((2,1)))
        dN3_dxy_GP4=np.matmul(Jinv, np.array([dN3_dsy[0][3],dN3_dsy[1][3]]).reshape((2,1)))
        dN4_dxy_GP4=np.matmul(Jinv, np.array([dN4_dsy[0][3],dN4_dsy[1][3]]).reshape((2,1)))
        
        dUxdx_GP4= dN1_dxy_GP4[0][0]*UxN1+ dN2_dxy_GP4[0][0]*UxN2+ dN3_dxy_GP4[0][0]*UxN3+ dN4_dxy_GP4[0][0]*UxN4
        dUxdy_GP4= dN1_dxy_GP4[1][0]*UxN1+ dN2_dxy_GP4[1][0]*UxN2+ dN3_dxy_GP4[1][0]*UxN3+ dN4_dxy_GP4[1][0]*UxN4
        
        dUydx_GP4= dN1_dxy_GP4[0][0]*UyN1+ dN2_dxy_GP4[0][0]*UyN2+ dN3_dxy_GP4[0][0]*UyN3+ dN4_dxy_GP4[0][0]*UyN4
        dUydy_GP4= dN1_dxy_GP4[1][0]*UyN1+ dN2_dxy_GP4[1][0]*UyN2+ dN3_dxy_GP4[1][0]*UyN3+ dN4_dxy_GP4[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP4= dUxdx_GP4
        e_yy_GP4= dUydy_GP4
        e_xy_GP4= 0.5*(dUydx_GP4+dUxdy_GP4)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP4= (self.E*(e_xx_GP4+ self.nu*e_yy_GP4)/(1-self.nu**2))*density_SIMP        
        S_yy_GP4= (self.E*(e_yy_GP4+ self.nu*e_xx_GP4)/(1-self.nu**2))*density_SIMP
        S_xy_GP4= (self.E*e_xy_GP4/(1+self.nu))*density_SIMP
        
        strainEnergy_GP4= (e_xx_GP4*S_xx_GP4+ e_yy_GP4*S_yy_GP4+ 2*e_xy_GP4*S_xy_GP4)

        # Strain energy at element
        strainEnergy_at_elem = ( strainEnergy_GP1 +strainEnergy_GP2 +strainEnergy_GP3 +strainEnergy_GP4 ) * detJ * 0.5
        

        return torch.sum( strainEnergy_at_elem )