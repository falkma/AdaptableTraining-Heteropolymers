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

#threshold (in kT) for displaying interaction differences
threshold = 1.

#we shuffle the polymer indices so that the ends are displayed in the center of the matrix
#this a purely cosmetic choice for more easily seeing where things are changing
order = np.roll(np.arange(N),N//2)

li = np.tril_indices(N,-2)
collection = []
mins = []
maxs = []

for i in np.arange(len(distributions)):

    #now we average over all matrices in an ensemble
    avg = np.mean(distributions[i],axis=0)

    #take out the diagonal since those terms don't have meaning
    np.fill_diagonal(avg,np.nan)

    #same for the two off-diagonals, since those interactions are set by the harmonic bonds
    od = np.arange(len(avg)-1)
    avg[od,od+1] = np.nan
    avg[od+1,od] = np.nan

    #shuffle so the ends of the polymer are in the center
    avg_shuffle = avg[order,:]
    avg_shuffle = avg_shuffle[:,order]

    collection.append(avg_shuffle)

diff_collection = []
maxs = []
for i in np.arange(len(folder)):

    outline1 = collection[2*i]
    outline2 = collection[2*i+1]    

    diff = outline1-outline2

    #threshold so that only differences of > 1 kT are shown
    mask = (1+np.sign(np.abs(diff)-threshold))/2
    mask = np.nan_to_num(mask)

    diff_collection.append(mask*diff)
    maxs.append(np.nanmax(mask*diff))

np.save('oscillating_training_diff_collection.npy',diff_collection[0])
np.save('singletarget_training_diff_collection.npy',diff_collection[1])

fig, axes = plt.subplots(nrows=len(folder), ncols=1)

i=0
for ax in axes.flat:

    im = ax.imshow(np.abs(diff_collection[i]),cmap = 'Greys',vmin = 0 ,vmax = np.max(maxs) )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    i += 1

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('training_compare_differences.pdf',format = 'pdf',bbox_inches="tight")
plt.savefig('training_compare_differences.png',format = 'png',bbox_inches="tight")

plt.show()
