import numpy as np
import hoomd
import hoomd.md
import scipy.spatial
import sys
import matplotlib.pyplot as plt
hoomd.context.initialize("");

run = int(sys.argv[1])

folder = 'adaptable_configurations4_c%d' % run

def compute_adjacency(positions,r):

    dist = scipy.spatial.distance.cdist(positions,positions)
    target_adj = np.sign(r-dist)
    target_adj = (1+target_adj)/2

    return target_adj.astype(np.int)

def compute_reward(p,t):

    if t == 1:
        p = p-p[0]
        theta = np.arctan2(p[1:,0],p[1:,1])
        d_theta = np.diff(theta)
        d_theta = np.where(d_theta > np.pi, d_theta - 2*np.pi, d_theta)
        d_theta = np.where(d_theta < -np.pi, d_theta + 2*np.pi, d_theta)
        reward = np.abs(np.sum(d_theta))

    if t == -1:
        p = np.flip(p,0)
        p = p-p[0]
        theta = np.arctan2(p[1:,0],p[1:,1])
        d_theta = np.diff(theta)
        d_theta = np.where(d_theta > np.pi, d_theta - 2*np.pi, d_theta)
        d_theta = np.where(d_theta < -np.pi, d_theta + 2*np.pi, d_theta)
        reward = np.abs(np.sum(d_theta))

    return reward/2/np.pi

def run_simulation(initial_positions,sim_time):

    snapshot = system.take_snapshot()
    snapshot.particles.position[:] =  initial_positions
    system.restore_snapshot(snapshot)

    #HOW LONG TO RUN
    hoomd.run(sim_time)
    p = system.take_snapshot().particles.position

    return p

###############################################################
#setting up simulation

N = 13
T = 1

positions = np.zeros((N,3))
dim = np.round(np.sqrt(N))
for i in np.arange(N):

    positions[i] = np.array([1.005*i,0,0])

positions -= np.mean(positions,axis = 0)
init_pos = np.copy(positions)
np.save(folder+'_initial_positions.npy',init_pos)

r = 1.1*np.max(np.linalg.norm(positions,axis=1))

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

Cd = np.load('../data_folder/oscillating_training_C2dist.npy')
C = Cd[run]

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
hoomd.md.integrate.langevin(group=all, kT=T, dscale = 5, seed=np.random.randint(100000));
hoomd.option.set_notice_level(0)

print('simulation set!')

#################################################################

#Training loop
num_episodes = 1
snapshot_init = system.take_snapshot(all=True)
frames = dict()
value = dict()

fin_pos = np.zeros((num_episodes,N,3))
values = np.zeros(num_episodes)

for time in np.arange(num_episodes):

    print('********')

    positions = run_simulation(init_pos,100000)
    values[time] = compute_reward(positions,-1)
    fin_pos[time] = positions

    if time%100 == 0:

        np.save(folder+'_history.npy',values)
        np.save(folder+'_final_positions.npy',fin_pos)

    print(time)
    print(values[time])

np.save(folder+'_history.npy',values)
np.save(folder+'_final_positions.npy',fin_pos)
