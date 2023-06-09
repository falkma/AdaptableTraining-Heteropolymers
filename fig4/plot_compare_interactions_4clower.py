import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"

folder = ['../data_folder/oscillating_training', '../data_folder/singletarget_training']

distributions = []

#load in the ensemble for the two types of training
for i in np.arange(len(folder)):

    distributions.append(np.load(folder[i]+'_C1dist.npy'))
    distributions.append(np.load(folder[i]+'_C2dist.npy'))

#length of polymer
N = 13

fig, axes = plt.subplots(nrows=len(folder), ncols=2)

#we shuffle the polymer indices so that the ends are displayed in the center of the matrix
#this a purely cosmetic choice for more easily seeing where things are changing
order = np.roll(np.arange(N),N//2)

li = np.tril_indices(N,-2)
collection = []

#set upper and lower bounds (in kT)
lb = 0.
ub = 6.

i=0
for ax in axes.flat:

    #now we average over all matrices in an ensemble
    avg = np.mean(distributions[i],axis=0)

    #take out the diagonal since those terms don't have physical relevance
    np.fill_diagonal(avg,np.nan)

    #same for the two off-diagonals, since those interactions are set by the harmonic bonds
    od = np.arange(len(avg)-1)
    avg[od,od+1] = np.nan
    avg[od+1,od] = np.nan

    #we record only those interactions
    #which are on average higher than the 75th percentile of the averaged
    #interaction matrix - these are presumably the "important contacts"
    mask = (1+np.sign(avg-np.nanpercentile(avg,75)))/2
    mask_shuffle = mask[order,:]
    mask_shuffle = mask_shuffle[:,order]

    #shuffle so the ends of the polymer are in the center
    avg_shuffle = avg[order,:]
    avg_shuffle = avg_shuffle[:,order]
    
    #make the matrix half and half the 75th-percentile stencil
    #and also the actual data
    p_mat = np.ceil(avg_shuffle*mask_shuffle)*ub
    p_mat[li] = avg_shuffle[li]

    im = ax.imshow(p_mat.T,cmap = 'Greys',vmin = lb,vmax = ub )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    i += 1
    collection.append(avg)


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('training_compare_interactions.png',format = 'png',bbox_inches="tight")
plt.savefig('training_compare_interactions.pdf',format = 'pdf',bbox_inches="tight")

plt.show()
