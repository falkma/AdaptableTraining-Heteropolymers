import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["font.sans-serif"] = "Arial"

folder = '../data_folder/oscillating_training/oscillation39'
trials = 0
window = 1

values = np.array(np.load(folder+'_c%d_history.pickle' % trials,allow_pickle=True))
alt_values = np.array(np.load(folder+'_c%d_alt_history.pickle' % trials,allow_pickle=True))

plt.figure(figsize=(6,2))
plt.plot(np.arange(len(values))/5,alt_values/1.0,color='blue',label='G1')
plt.plot(np.arange(len(values))/5,values/1.0,color='red',label='G2')
plt.xticks(np.arange(11),np.arange(11))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('cost function')

plt.savefig('oscillating_training_trajectory.pdf',format = 'pdf',bbox_inches="tight")
plt.savefig('oscillating_training_trajectory.png',format = 'png',bbox_inches="tight")

plt.show()

