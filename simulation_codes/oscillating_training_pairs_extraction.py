import numpy as np
import pickle

#number of training trajectories
run_max = 200

#how many training epochs each time a goal gets trained for
period = 5

#number of switches between the two goals
osc_num = 10

#polymer length
N = 13

#window in a given training period over which we look for a succesfully trained interaction matrix
window = 3

#initializing storage lists
C1_dist = []
C2_dist = []
C1_pos = []
C2_pos = []

runs = np.arange(run_max)

#set name of which type of training for file prefix
folder = 'oscillating_training'

#this block of code iterates through all training runs
#if an interaction matrix scores below -.69 in the last 3-epoch window of a 5-epoch period
#then it is considered a succesfully trained matrix. (-.7 is the ideal minimum score).
#If multiple such matrices are recorded, we take the one with the lowest score.
#If two such matrices are recorded in consecutive periods, these are 
#succesfully adaptable matrices, and we add them to the ensemble in the storage lists.
success = 0
for run in runs:

    #load in data (score, interaction matrix, positions)
    trial_folder = folder+'_c%d' % run
    with open(trial_folder+'_value.pickle', 'rb') as handle:
        values = pickle.load(handle)
    with open(trial_folder+'_frames.pickle', 'rb') as handle:
        sequences = pickle.load(handle)
    with open(trial_folder+'_positions.pickle', 'rb') as handle:
        temp_pos = pickle.load(handle)
    keys = list(sequences)

    #now iterate over all periods (here, 5 epochs)
    counter = 0
    for i in np.arange(osc_num-1):

        temp1 = np.zeros((2,window))
        temp2 = np.zeros((2,window))

        #look inside the last window of a period (here, 3 epochs)
        for j in np.arange(window):

            t1 = int((i+1)*period-window+j)
            temp1[0][j] = t1
            temp1[1][j] = values[keys[t1]]
            t2 = int((i+2)*period-window+j)
            temp2[0][j] = t2
            temp2[1][j] = values[keys[t2]]

        #take the best scoring interaction matrix and record it, 
        #the time it occurs, and the value it reaches.
        i1 = np.argmin(temp1[1])
        i2 = np.argmin(temp2[1])
        time1 = int(temp1[0][i1])
        time2 = int(temp2[0][i2])
        val1 = temp1[1][i1]
        val2 = temp2[1][i2]

        #if consecutive scores are both good, record the matrix in the storage lists
        if val1 < -.69 and val2 < -.69:

            counter += 1

            seq1 = sequences[keys[time1]]
            C1 = np.zeros((N,N))
            ui = np.triu_indices(len(C1),2)
            C1[ui] = seq1
            C1.T[ui] = C1[ui]

            seq2 = sequences[keys[time2]]
            C2 = np.zeros((N,N))
            ui = np.triu_indices(len(C2),2)
            C2[ui] = seq2
            C2.T[ui] = C2[ui]

            #need to distinguish between matrices good for goal 1 vs goal 2
            if i%2 == 1:
                C1_dist.append(C1)
                C2_dist.append(C2)
                C1_pos.append(temp_pos[keys[time1]])
                C2_pos.append(temp_pos[keys[time2]])

            else:
                C2_dist.append(C1)
                C1_dist.append(C2)
                C2_pos.append(temp_pos[keys[time1]])
                C1_pos.append(temp_pos[keys[time2]])

    success += min(1, counter)

Cpairs_dist = []
Cpairs_dist.append(C1_dist)
Cpairs_dist.append(C2_dist)

#save the ensemble of succesfully adaptable matrices
np.save(folder+'_C1dist.npy',C1_dist)
np.save(folder+'_C2dist.npy',C2_dist)