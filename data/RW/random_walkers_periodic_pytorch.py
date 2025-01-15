import numpy as np
import torch

def get_density(cell_centers, pos):
    # Evaluating density of particles
    # pos should be periodic boundary shifted
    ncells = torch.numel(cell_centers)
    num_par = torch.numel(pos)
    dx = cell_centers[1]-cell_centers[0]
    cell_centers = cell_centers.unsqueeze(0)
    cell_centers = cell_centers.expand((num_par,-1))
    pos = pos.unsqueeze(-1)
    pos = pos.expand((-1,ncells))
    # Get relative distance from cell centers
    rel_dist = cell_centers - pos 
    dens_i_1 = torch.abs(rel_dist) < 0.5*dx
    dens_i_2 = rel_dist == -0.5*dx
    dens_i = dens_i_1 + dens_i_2
    # Sum along dim = 0
    density = torch.sum(dens_i,dim=0)
    return density

def get_flux(face_centers, pos_0, jump):
    # Evaluating flux at faces
    num_par = torch.numel(pos_0)
    nfaces = torch.numel(face_centers)
    flux = torch.zeros(nfaces-1)
    face_centers = face_centers.unsqueeze(0)
    face_centers = face_centers.expand((num_par,-1))

    # Do NOT apply periodic shift
    pos_1 = pos_0 + jump
    # Get position at step i
    pos_0 = pos_0.unsqueeze(-1)
    pos_0 = pos_0.expand((-1,nfaces))

    pos_1 = pos_1.unsqueeze(-1)
    pos_1 = pos_1.expand((-1,nfaces))

    # particles going left to right
    cond_1_1 = (face_centers-pos_0) >= 0.
    cond_1_2 = (face_centers-pos_1) <= 0.
    # Logical AND
    cond_1 = torch.logical_and(cond_1_1,cond_1_2)
    flux_Left_Right = torch.sum(cond_1,dim=0)

    # particles going right to left
    cond_2_1 = (face_centers-pos_0) <= 0.
    # EQUAL TO SIGN SHOULD NOT APPEAR
    cond_2_2 = (face_centers-pos_1) > 0.
    # Logical AND
    cond_2 = torch.logical_and(cond_2_1,cond_2_2)
    flux_Right_Left = torch.sum(cond_2,dim=0)

    flux[1:] = flux_Left_Right[1:-1] - flux_Right_Left[1:-1]
    # For the periodic face, sum both ends
    flux[0] = (flux_Left_Right[0] - flux_Right_Left[0]
              + flux_Left_Right[-1] - flux_Right_Left[-1])
    return flux

def random_walk_just_evolve(ncells, nmoves, dt, initial_pos,len_system = 1.0):
    # total number of particles
    num_par = torch.numel(initial_pos)
    # std of jump
    jump_std = torch.tensor(dt)
    jump_std = torch.sqrt(jump_std)
    # ASSUMING SYSTEM IS OF UNIT LENGTH
    dx = len_system/ncells
    # Particles cannot jump more than dx
    jump_lim = torch.tensor(dx)
    # Temporally evolving particle positions
    pos = initial_pos
    for i in range(1,nmoves+1):
        jump = jump_std*torch.randn(num_par)
        # Clamp the jump to cell size
        jump = torch.clamp(jump,min=(-1+1e-6)*jump_lim,max=(1-1e-6)*jump_lim)
        # Update particle position
        new_pos = pos + jump
        # Apply periodic effect
        # lower bound
        new_pos = torch.where(new_pos < 0., new_pos + len_system, new_pos)
        # upper bound
        new_pos = torch.where(new_pos >= len_system, new_pos - len_system, new_pos)
        pos = new_pos
    return pos

def random_walk(ncells, nmoves, dt, initial_pos,len_system = 1.0):
    # total number of particles
    num_par = torch.numel(initial_pos)
    # Tensor keep track of particle positions
    particles = torch.zeros((nmoves+1,num_par))
    # Tensor to keep track of density of particles
    density = torch.zeros((nmoves+1,ncells))
    # Tensor to keep track of interior faces flux
    flux = torch.zeros((nmoves,ncells))
    # Initial position of particles
    # ASSUMING SYSTEM IS OF UNIT LENGTH
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,(ncells-0.5)*dx,ncells)
    face_centers = torch.linspace(0,len_system,ncells+1)
    particles[0,:] = initial_pos
    density[0,:] = get_density(cell_centers,particles[0,:]) 

    # std of jump
    jump_std = torch.tensor(dt)
    jump_std = torch.sqrt(jump_std)
    # Particles cannot jump more than dx
    jump_lim = torch.tensor(dx)
    # Temporally evolving particle positions
    for i in range(1,nmoves+1):
        jump = jump_std*torch.randn(num_par)
        # Clamp the jump to cell size
        jump = torch.clamp(jump,min=(-1+1e-6)*jump_lim,max=(1-1e-6)*jump_lim)
        pos = particles[i-1,:]
        # Leverage pos and jump to get flux
        flux[i-1,:] = get_flux(face_centers,pos,jump)
        # Update particle position
        new_pos = pos + jump
        # Apply periodic effect
        # lower bound
        new_pos = torch.where(new_pos < 0., new_pos + len_system, new_pos)
        # upper bound
        new_pos = torch.where(new_pos >= len_system, new_pos - len_system, new_pos)
        particles[i,:] = new_pos
        # Get density
        density[i,:] = get_density(cell_centers,new_pos)

    return particles, density, flux

# Always uniformly distributing at cell centers
def get_initial_pos(ncells,par_per_cell,len_system=1.0):
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    cell_centers = cell_centers.unsqueeze(-1)
    cell_centers = cell_centers.expand((-1,par_per_cell))
    cell_centers = torch.reshape(cell_centers,(-1,))
    return cell_centers

if __name__ == "__main__":
    torch.set_default_device('cuda')
    par_per_cell = 50
    ncells = 100
    num_par = par_per_cell*ncells
    nmoves = 100
    len_system=1.0
    dx = len_system/ncells
    cell_centers = torch.linspace(0.5*dx,len_system-0.5*dx,ncells)
    initial_pos = get_initial_pos(ncells,par_per_cell,
                                  len_system=len_system)
    particles = random_walk_just_evolve(ncells,nmoves,1.0,
                                          initial_pos,
                                          len_system=len_system)
    print(particles)
    density = get_density(cell_centers,particles)
    print(density)
    print(torch.sum(density))
