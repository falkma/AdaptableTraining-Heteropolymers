import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.signal

#Estmate barrier contacts in energy landscape using peak prominence

def define_target_structure(configuration,N,N_total,t):

    if t == 1:
        target = configuration[0:N]
    else:
        target = configuration[N_total-N:N_total]

    dist = scipy.spatial.distance.cdist(target,target)
    target_adj = np.sign(1.4-dist)
    target_adj = (1+target_adj)/2
    total_adj = np.zeros((N_total,N_total))
    if t == 1:
        total_adj[:N,:N] = target_adj.astype(np.int)
    else:
        total_adj[N_total-N:,N_total-N:] = target_adj.astype(np.int)

    return total_adj, target

def sticking_energy(B,C):

    n = len(B)
    np.fill_diagonal(B,0)
    od = np.arange(n-1)
    B[od,od+1] = np.zeros(n-1)
    B[od+1,od] = np.zeros(n-1)

    return -np.sum(np.multiply(B,C))/2

def bending_energy(positions,k):

    if len(positions) < 2:
        E = 0
    else:
        t = positions[0:-1,:]-positions[1:,:]
        E = 0

        for i in np.arange(len(t)-1):

            theta = np.arccos(np.dot(-t[i],t[i+1])/np.linalg.norm(t[i])/np.linalg.norm(t[i+1]))
            E += .5*k*(theta-np.pi)**2

    return E

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

N = 13
order = np.roll(np.arange(N),N//2)
change_matrix = np.zeros((N,N))

combined_energy = []
combined_energy_alt = []

for index in 1+np.arange(2):

    Cd = np.load('../data_folder/oscillating_training_C%ddist.npy' % index)
    trials = 40

    E = np.zeros((trials,N))
    Eatt = np.zeros((trials,N))
    Eben = np.zeros((trials,N))
    E_alt = np.zeros((trials,N))
    Eatt_alt = np.zeros((trials,N))
    Eben_alt = np.zeros((trials,N))
    ontarget = []
    offtarget = []

    partn = np.zeros((trials,N))
    partn_alt = np.zeros((trials,N))

    for t in np.arange(trials):

        C = Cd[t]
        configuration = np.load('../data_folder/oscillating_training_C1configs/adaptable_configurations3_c%d_final_positions.npy' % t)[0]
        configuration_alt = np.load('../data_folder/oscillating_training_C2configs/adaptable_configurations4_c%d_final_positions.npy' % t)[0]

        for i in np.arange(N):

            adj, target = define_target_structure(configuration,i+1,N,1)
            adj_alt, target_alt = define_target_structure(configuration_alt,i+1,N,-1)

            partn[t,i] = compute_reward(target,1)
            partn_alt[t,i] = compute_reward(target_alt,-1)
            
            Eatt[t,i] = sticking_energy(adj,C)
            Eben[t,i] = bending_energy(target,5.)
            E[t,i] = Eben[t,i] + Eatt[t,i]

            Eatt_alt[t,i] = sticking_energy(adj_alt,C)
            Eben_alt[t,i] = bending_energy(target_alt,5.)
            E_alt[t,i] = Eben_alt[t,i] + Eatt_alt[t,i]  

        goalzone = np.where(partn[t] > .99)[0]
        goalzone_alt = np.where(partn_alt[t] > .99)[0]

        ontarget.append(np.min(E[t][goalzone]))
        offtarget.append(np.min(E_alt[t][goalzone_alt]))

    barrier_adj = []
    barrier_adj_alt = []

    for t in np.arange(trials):

        configuration = np.load('../data_folder/oscillating_training_C1configs/adaptable_configurations3_c%d_final_positions.npy' % t)[0]
        configuration_alt = np.load('../data_folder/oscillating_training_C2configs/adaptable_configurations4_c%d_final_positions.npy' % t)[0]

        barriers = scipy.signal.find_peaks(E[t],prominence=1)[0]
        if len(barriers) > 0:
            barrier = 1+np.max(barriers)
        else:
            barrier = 0
        
        barriers_alt = scipy.signal.find_peaks(E_alt[t],prominence=1)[0]
        if len(barriers_alt) > 0:
            barrier_alt = 1+np.max(barriers_alt)
        else:
            barrier_alt = 0

        adj, target = define_target_structure(configuration,barrier+1,N,1)
        adj_alt, target_alt = define_target_structure(configuration_alt,barrier_alt+1,N,-1)

        barrier_adj.append(adj)
        barrier_adj_alt.append(adj_alt)

    barrier_adj = np.array(barrier_adj)
    barrier_adj_alt = np.array(barrier_adj_alt)

    if index%2 == 1:
        adj = np.mean(barrier_adj_alt,axis=0)
    else:
        adj = np.mean(barrier_adj,axis=0) 
    np.fill_diagonal(adj,np.nan)
    od = np.arange(N-1)
    adj[od,od+1] = np.nan
    adj[od+1,od] = np.nan
    adj_shuffle = adj[order,:]
    adj_shuffle = adj_shuffle[:,order]
    change_matrix += adj_shuffle/2

cutoff = np.nanpercentile(change_matrix[np.nonzero(change_matrix)],75)
masked_data = np.sign(np.ma.masked_where(np.abs(change_matrix) < cutoff, np.abs(change_matrix)))

diff_collection = np.load('../fig4/oscillating_training_diff_collection.npy')
plt.figure()
plt.imshow(diff_collection,cmap='coolwarm',vmax=3,vmin=-3)
plt.imshow(masked_data,cmap='Greys',vmax=1,vmin=0,alpha=.85)
plt.savefig('oscillating_training_overall_barrier_contacts2.pdf' % index,format = 'pdf',bbox_inches="tight")
plt.savefig('oscillating_training_overall_barrier_contacts2.png' % index,format = 'png',bbox_inches="tight")

diff_collection = np.load('../fig4/singletarget_training_diff_collection.npy')
plt.figure()
plt.imshow(diff_collection,cmap='coolwarm',vmax=3,vmin=-3)
plt.imshow(masked_data,cmap='Greys',vmax=1,vmin=0,alpha=.85)
plt.savefig('singletarget_training_overall_barrier_contacts2.pdf' % index,format = 'pdf',bbox_inches="tight")
plt.savefig('singletarget_training_overall_barrier_contacts2.png' % index,format = 'png',bbox_inches="tight")

plt.close()