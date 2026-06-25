import torch
import numpy as np
import json
import random

def set_seed0(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)

def load_job_tensors_from_json(file_path, job_name):
    """
    Loads tensors for a specific job from a JSON file and converts them back to PyTorch tensors.
    
    Args:
    - file_path (str): Path to the JSON file.
    - job_name (str): The name of the job to extract tensors for (e.g., 'Job-1').

    Returns:
    - dict: A dictionary where keys are tensor names and values are PyTorch tensors.
    """
    with open(file_path, 'r') as json_file:
        jobs_dict_serializable = json.load(json_file)
    
    # Extract the dictionary for the specific job
    if job_name in jobs_dict_serializable:
        job_tensors_serializable = jobs_dict_serializable[job_name]
        # Convert lists back to tensors
        job_tensors = {key: torch.tensor(value) for key, value in job_tensors_serializable.items()}
        return job_tensors
    else:
        raise ValueError(f"No tensors found for {job_name}")

def get_tkwargs():
    """
    Returns a dictionary with the device and dtype configuration
    for PyTorch, prioritizing CUDA, then MPS (Mac GPU), and finally CPU.
    """
    # Determine the device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cpu")  # Use NVIDIA GPU if available
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")  # Use Mac GPU (MPS) if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    
    # Define tkwargs with selected device and dtype
    tkwargs = {
        "dtype": torch.float32,
        "device": device,
    }
    
    print(f"Using device: {device}")
    return tkwargs 

def central_diff_2D_2nd(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with second-order accuracy
    df_dx = (torch.roll(f, shifts=-1, dims=1) - torch.roll(f, shifts=1, dims=1)) / (2 * dx)
    df_dy = (torch.roll(f, shifts=-1, dims=0) - torch.roll(f, shifts=1, dims=0)) / (2 * dy)
    
    # Second-order derivatives with second-order accuracy
    d2f_dx2 = (torch.roll(f, shifts=-1, dims=1) - 2 * f + torch.roll(f, shifts=1, dims=1)) / (dx ** 2)
    d2f_dy2 = (torch.roll(f, shifts=-1, dims=0) - 2 * f + torch.roll(f, shifts=1, dims=0)) / (dy ** 2)
    
    # Mixed second derivative with second-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)) / (4 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def central_diff_2D_4th(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with fourth-order accuracy
    df_dx = (-torch.roll(f, shifts=-2, dims=1) 
             + 8 * torch.roll(f, shifts=-1, dims=1)
             - 8 * torch.roll(f, shifts=1, dims=1) 
             + torch.roll(f, shifts=2, dims=1)) / (12 * dx)
    
    df_dy = (-torch.roll(f, shifts=-2, dims=0) 
             + 8 * torch.roll(f, shifts=-1, dims=0)
             - 8 * torch.roll(f, shifts=1, dims=0) 
             + torch.roll(f, shifts=2, dims=0)) / (12 * dy)
    
    # Second-order derivatives with fourth-order accuracy
    d2f_dx2 = (-torch.roll(f, shifts=-2, dims=1) 
               + 16 * torch.roll(f, shifts=-1, dims=1)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=1) 
               - torch.roll(f, shifts=2, dims=1)) / (12 * dx**2)
    
    d2f_dy2 = (-torch.roll(f, shifts=-2, dims=0) 
               + 16 * torch.roll(f, shifts=-1, dims=0)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=0) 
               - torch.roll(f, shifts=2, dims=0)) / (12 * dy**2)
    
    # Mixed second derivative with fourth-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-2, dims=1)
                - torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=2, dims=1)
                + torch.roll(torch.roll(f, shifts=2, dims=0), shifts=2, dims=1)) / (144 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def central_diff_3D_2nd(f, dx, dy, dz):
    # the higher order numerical difference is not necessary
    """
    Compute first and second derivatives of a 3D tensor using central difference.
    
    Parameters:
    - f: Input 3D tensor (Ny, Nx, Nz).
    - dx, dy, dz: Grid spacings along x, y, and z directions.
    
    Returns:
    - A list of first and second derivatives: 
      [df_dx, df_dy, df_dz, d2f_dx2, d2f_dy2, d2f_dz2, d2f_dxdy, d2f_dxdz, d2f_dydz]
    """

    # First-order derivatives
    df_dx = (torch.roll(f, shifts=-1, dims=1) - torch.roll(f, shifts=1, dims=1)) / (2 * dx)  # x is dim=1
    df_dy = (torch.roll(f, shifts=-1, dims=0) - torch.roll(f, shifts=1, dims=0)) / (2 * dy)  # y is dim=0
    df_dz = (torch.roll(f, shifts=-1, dims=2) - torch.roll(f, shifts=1, dims=2)) / (2 * dz)  # z is dim=2

    # Second-order derivatives
    d2f_dx2 = (torch.roll(f, shifts=-1, dims=1) - 2 * f + torch.roll(f, shifts=1, dims=1)) / (dx ** 2)  # x is dim=1
    d2f_dy2 = (torch.roll(f, shifts=-1, dims=0) - 2 * f + torch.roll(f, shifts=1, dims=0)) / (dy ** 2)  # y is dim=0
    d2f_dz2 = (torch.roll(f, shifts=-1, dims=2) - 2 * f + torch.roll(f, shifts=1, dims=2)) / (dz ** 2)  # z is dim=2

    # Mixed second derivatives
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)) / (4 * dx * dy)

    d2f_dxdz = (torch.roll(torch.roll(f, shifts=-1, dims=2), shifts=-1, dims=1)
                - torch.roll(torch.roll(f, shifts=-1, dims=2), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=1, dims=2), shifts=-1, dims=1)
                + torch.roll(torch.roll(f, shifts=1, dims=2), shifts=1, dims=1)) / (4 * dx * dz)

    d2f_dydz = (torch.roll(torch.roll(f, shifts=-1, dims=2), shifts=-1, dims=0)
                - torch.roll(torch.roll(f, shifts=-1, dims=2), shifts=1, dims=0)
                - torch.roll(torch.roll(f, shifts=1, dims=2), shifts=-1, dims=0)
                + torch.roll(torch.roll(f, shifts=1, dims=2), shifts=1, dims=0)) / (4 * dy * dz)

    return [df_dx, df_dy, df_dz, d2f_dx2, d2f_dy2, d2f_dz2, d2f_dxdy, d2f_dxdz, d2f_dydz]

def central_diff_3D_2nd_simplified(f, dx, dy, dz):
    # the higher order numerical difference is not necessary
    """
    Compute first and second derivatives of a 3D tensor using central difference.
    
    Parameters:
    - f: Input 3D tensor (Ny, Nx, Nz).
    - dx, dy, dz: Grid spacings along x, y, and z directions.
    
    Returns:
    - A list of first and second derivatives: 
      [df_dx, df_dy, df_dz, d2f_dx2, d2f_dy2, d2f_dz2, d2f_dxdy, d2f_dxdz, d2f_dydz]
    """

    # First-order derivatives
    df_dx = (torch.roll(f, shifts=-1, dims=1) - torch.roll(f, shifts=1, dims=1)) / (2 * dx)  # x is dim=1
    df_dy = (torch.roll(f, shifts=-1, dims=0) - torch.roll(f, shifts=1, dims=0)) / (2 * dy)  # y is dim=0
    df_dz = (torch.roll(f, shifts=-1, dims=2) - torch.roll(f, shifts=1, dims=2)) / (2 * dz)  # z is dim=2

    return [df_dx, df_dy, df_dz]

def projectDensity(x,b=16):
    nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
    return x