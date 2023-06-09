import numpy as np
import pickle
import hoomd
import hoomd.md
import scipy.spatial
import itertools
import random
import sys
import cma
hoomd.context.initialize("");

run = 1*int(sys.argv[1])+int(sys.argv[2])

folder = 'oscillation_training_c%d' % run

def compute_reward(p,t):

    #computes reward given particle positions p and polymer orientation t

    if t == 1:
        #center first particle at origin
        p = p-p[0]

        #compute changes in winding
        theta = np.arctan2(p[1:,0],p[1:,1])
        d_theta = np.diff(theta)
        
        #account for any large jumps
        d_theta = np.where(d_theta > np.pi, d_theta - 2*np.pi, d_theta)
        d_theta = np.where(d_theta < -np.pi, d_theta + 2*np.pi, d_theta)

        #compute reward as magnitude of winding in theta
        reward = np.abs(np.sum(d_theta))

    if t == -1:

        #flip polymer so last particle is centered at origin before computing reward
        p = np.flip(p,0)
        p = p-p[0]
        theta = np.arctan2(p[1:,0],p[1:,1])
        d_theta = np.diff(theta)
        d_theta = np.where(d_theta > np.pi, d_theta - 2*np.pi, d_theta)
        d_theta = np.where(d_theta < -np.pi, d_theta + 2*np.pi, d_theta)
        reward = np.abs(np.sum(d_theta))

    #return normalized reward
    return reward/2/np.pi

def run_simulation(initial_positions,seq,sim_time):

    #set simulation positions to a particular initial configuration
    snapshot = system.take_snapshot()
    snapshot.particles.position[:] =  initial_positions
    system.restore_snapshot(snapshot)

    #fill in matrix of interactions
    C = np.zeros((len(initial_positions),len(initial_positions)))
    ui = np.triu_indices(len(C),2)
    C[ui] = seq
    C.T[ui] = C[ui]

    #add in morse interactions
    for i in np.arange(len(C)):
        for j in np.arange(len(C)):
            morse.pair_coeff.set(colors[i], colors[j], D0=C[i,j], alpha=6, r0=1.1)

    #run simulation
    hoomd.run(sim_time)
    positions = system.take_snapshot().particles.position

    return positions, C

def cost_function(init_pos,child_sequence,target,alt_target,samples):

    #compute cost function of an interaction matrix averaged over multiple simulation replicates
    values = np.zeros(samples)
    alt_values = np.zeros(samples)

    for i in np.arange(samples):
        positions, matint = run_simulation(init_pos,child_sequence,100000)
        values[i] = -compute_reward(positions,target)
        alt_values[i] = -compute_reward(positions,alt_target)

    #clip reward so that maximum achievable target is 70%
    reward = -np.clip(len(np.where(values < -1.0)[0])/np.float(samples),0,.7)
    alt_reward = -np.clip(len(np.where(alt_values < -1.0)[0])/np.float(samples),0,.7)

    return positions, reward, matint, alt_reward

###############################################################
#setting up simulation

#length of polymer
N = 13

#value of kT
T = 1

positions = np.zeros((N,3))
dim = np.round(np.sqrt(N))

#initial configuration is a straight line
for i in np.arange(N):

    positions[i] = np.array([1.005*i,0,0])

positions -= np.mean(positions,axis = 0)
init_pos = np.copy(positions)
np.save(folder+'_initial_positions.npy',init_pos)

r = 2.0*np.max(np.linalg.norm(positions,axis=1))

lx = r+1
ly = r+1
lz = r+1

x_boxdim = 2.*lx
y_boxdim = 2.*ly
z_boxdim = 2.*lz

colors = [str(i) for i in np.arange(N)]
color_num = len(colors)

snapshot = hoomd.data.make_snapshot(N=N, box=hoomd.data.boxdim(Lx=x_boxdim, Ly=y_boxdim, dimensions=2),particle_types=colors,
           bond_types=['polymer'],angle_types=['polymer'])
snapshot.particles.position[:] =  positions
snapshot.particles.typeid[:] = np.arange(N)

snapshot.bonds.resize(N-1)
snapshot.bonds.group[:] = [[j,j+1] for j in np.arange(N-1)]
snapshot.bonds.typeid[:] = 0

snapshot.angles.resize(N-2)
snapshot.angles.group[:] = [[j,j+1,j+2] for j in np.arange(N-2)]
snapshot.angles.typeid[:] = 0

system = hoomd.init.read_snapshot(snapshot);
nl = hoomd.md.nlist.cell()

harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k=100.0*T, r0=1)

lp = 5.
stiffness = hoomd.md.angle.harmonic()
stiffness.angle_coeff.set('polymer', k=lp*T, t0=np.pi)

lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)
lj.set_params(mode='shift')

morse = hoomd.md.pair.morse(r_cut=2.0, nlist=nl)

max_strength = 6.
target1 = 1
target2 = -1
C = np.random.uniform(0,max_strength,size=(color_num,color_num))

for i in np.arange(color_num):
    for j in np.arange(color_num):

        lj.pair_coeff.set(colors[i], colors[j], epsilon=1.0, sigma=1.0)
        morse.pair_coeff.set(colors[i], colors[j], D0=C[i,j], alpha=6, r0=1.1)

origin1 = [-lx,0,0]
origin2 = [lx,0,0]
origin3 = [0,-ly,0]
origin4 = [0,ly,0]

normal1 = [1,0,0]
normal2 = [-1,0,0]
normal3 = [0,1,0]
normal4 = [0,-1,0]

#assert walls to avoid configurations where polymer is split across bounding box
wallstructure = hoomd.md.wall.group()
wallstructure.add_plane(origin1,normal1,inside=True)
wallstructure.add_plane(origin2,normal2,inside=True)
wallstructure.add_plane(origin3,normal3,inside=True)
wallstructure.add_plane(origin4,normal4,inside=True)

wall_lj = hoomd.md.wall.force_shifted_lj(wallstructure, r_cut=2**(1/6))
for i in np.arange(color_num):
    wall_lj.force_coeff.set(colors[i], sigma=1.0,epsilon=1.0)

all = hoomd.group.all();

dt = .005
hoomd.md.integrate.mode_standard(dt=dt);
hoomd.md.integrate.langevin(group=all, kT=T, dscale = 5, seed=123);
hoomd.option.set_notice_level(0)

print('simulation set!')


#################################################################

#Training loop

#number of switches between the two goals - each goal gets trained for num_episodes/2 times
num_episodes = 10

#how many training epochs each time a goal gets trained for
period = 5
snapshot_init = system.take_snapshot(all=True)
frames = dict()
value = dict()
fin_pos = dict()

episode_history = []
alt_history = []

#how many different matrices to consider n each CMA-ES evaluation
batch_num = 20

#how many simulation replicates to run for each niteraction matrx
sample_num = 40

state_dim = int((N*N-N-2*(N-1))/2)

#initialize interaction matrix as 0 matrix
x0 = max_strength*np.zeros(state_dim)
es = cma.CMAEvolutionStrategy(x0, 2, {'bounds': [0, max_strength]})

time = 0
for ep in np.arange(num_episodes):

    for step in np.arange(period):

        print('********')

        solutions = es.ask(number=batch_num)
        temp_values = np.zeros(batch_num)
        alt_values = np.zeros(batch_num)
        temp_pos = np.zeros((batch_num,N,3))
        for i in np.arange(batch_num):
            energies = solutions[i]
            temp_pos[i], temp_values[i], _, alt_values[i] = cost_function(init_pos,energies,target1,target2,sample_num)
            print(temp_values[i])

        if ep%2 == 1:
            es.tell(solutions,temp_values)
            argmin = np.argmin(temp_values)
            value[str(int(time))] = np.copy(temp_values[argmin])
        else:
            es.tell(solutions,alt_values)
            argmin = np.argmin(alt_values)
            value[str(int(time))] = np.copy(alt_values[argmin])
    
        frames[str(int(time))] = np.copy(np.array(solutions[argmin]))
        fin_pos[str(int(time))] = np.copy(np.array(temp_pos[argmin]))

        if time%10 == 9:

            #record the best interaction matrix in the current CMA-ES batch
            with open(folder+'_frames.pickle', 'wb') as handle:
                pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #record the cost function value attained for the best interaction matrix
            with open(folder+'_value.pickle', 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #record the final positions of the last simulation replicate of the best interaction matrix
            with open(folder+'_positions.pickle', 'wb') as handle:
                pickle.dump(fin_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(time)
        episode_history.append(temp_values[argmin])
        alt_history.append(alt_values[argmin])        

        if len(episode_history) >= 5+1:
            print("****************Solved***************")
            print("Mean cumulative reward over 5 episodes:{:0.2f}" .format(
                np.mean(episode_history[-5:])))

        #record the history of the best cost function value attained currently in training
        with open(folder+'_history.pickle', 'wb') as handle:
            pickle.dump(episode_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #as well as the corresponding value of the off-target cost function for that interaction matrix
        with open(folder+'_alt_history.pickle', 'wb') as handle:
            pickle.dump(alt_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        time += 1
