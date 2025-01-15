import numpy as np
import torch
import h5py
#######################################################
####### Local imports ################################
from random_walkers_periodic_pytorch import get_initial_pos, get_density
from random_walkers_periodic_pytorch import random_walk_just_evolve, random_walk
#######################################################

dataset_name = "test"

assert dataset_name in ["train", "valid", "test"]

print("Creating "+dataset_name+" dataset")

dataset_dct = {"train" : [1024, 140],
               "valid" : [ 128, 640],
               "test"  : [ 128, 640]}

torch.set_default_device('cuda')
par_per_cell = 5
ncells = 256
num_par = par_per_cell*ncells
dt = 1.0
nmoves = 100000
len_system=1.0
dx = len_system/ncells
cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
initial_pos = get_initial_pos(ncells,par_per_cell,
                              len_system=len_system)

# Trying to achieve statistical equilibrium
initial_pos = random_walk_just_evolve(ncells,nmoves,dt,
                     initial_pos,len_system=len_system)

n_initial_cond = int(dataset_dct[dataset_name][0])
n_temp_steps   = int(dataset_dct[dataset_name][1])

# Saving the number density
final_data = np.zeros((n_initial_cond, n_temp_steps, ncells))

for i in range(n_initial_cond):
    # Evolve a few steps to remove temporal correlation
    # between two initial conditions
    initial_pos = random_walk_just_evolve(ncells,10,dt,
                     initial_pos,len_system=len_system)
    # Get the first density
    final_data[i,0,:] = get_density(cell_centers,initial_pos).cpu().numpy()

    for j in range(1,n_temp_steps):
        # Evolve the initial condition by 1 step
        new_pos, new_density, _ = random_walk(ncells, 1, dt, initial_pos,
                                            len_system = len_system)
        # Update the position and density
        final_data[i,j,:] = new_density[-1,:].cpu().numpy()
        initial_pos = new_pos[-1,:]

# Simple normalization of data
final_data = (final_data - par_per_cell)/par_per_cell

with h5py.File(dataset_name+".h5", mode="w") as f:
    f.create_dataset("x", data=final_data, dtype = np.float32)
    f.create_dataset("ppc", (par_per_cell,), dtype = 'i')
