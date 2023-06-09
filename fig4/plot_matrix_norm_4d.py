import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"

folder = ['../data_folder/singletarget_training', '../data_folder/oscillating_training']

distributionsL0 = []
distributionsL1 = []

threshold = 2.

for i in np.arange(len(folder)):

    C1dist = np.load(folder[i]+'_C1dist.npy')
    C2dist = np.load(folder[i]+'_C2dist.npy')

    tempL0 = []
    N = len(C1dist[0])
    li = np.tril_indices(N,-2)

    for j in np.arange(len(C1dist)):

        seq1 = C1dist[j][li]
        seq2 = C2dist[j][li]

        differences = len(np.where(np.abs(seq1-seq2) > threshold)[0])
        tempL0.append(differences/len(seq1))

    distributionsL0.append(tempL0)

plt.figure(figsize=(3,6))
tick_positions = .5*np.arange(len(folder))
labels = [ 'no oscillation', 'with oscillation']
violin_plot = plt.violinplot(distributionsL0,positions=tick_positions,showmeans=True)
for pc in violin_plot['bodies']:
    pc.set_facecolor('black')
for partname in ('cbars','cmins','cmaxes'):
    vp = violin_plot[partname]
    vp.set_edgecolor('black')

plt.xticks(tick_positions, labels, rotation=10)
plt.ylabel('fraction of changed interactions')
plt.ylim([0,.65])
plt.savefig('training_L0_matrix_norm.pdf',format = 'pdf',bbox_inches="tight")
plt.savefig('training_L0_matrix_norm.png',format = 'png',bbox_inches="tight")
plt.show()