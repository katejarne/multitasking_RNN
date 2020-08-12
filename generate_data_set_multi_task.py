#########################################
#                                       #
#      A Multitask "And/or"             #
#        data set generator             #
#    of samples with contex input       #
#     with adjutable parameters         #
#                                       #
#   Mit License C. Jarne V. 1.0 2020    #
#########################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from numpy.random import seed
start_time = time.time()

#def generate_trials(size): #No time loop
def generate_trials(size, mem_gap):#time loop
    #seed(1)
    seed(None)
    #mem_gap          = 20 # output reaction time
    first_in         = 60 #-25#time to start the first stimulus   #30 #60
    first_in_c       = 20
    stim_dur         = 20 #stimulus duration #20 #30
    stim_noise       = 0.1 #noise
    var_delay_length = 0#int(20*np.random.rand()) #20 #change for a variable length stimulus
    out_gap          = 140#+25#130 #how much lenth add to the sequence duration    #140 #100
    sample_size      = size # sample size
    rec_noise        = 0
    #print("var_delay_length",var_delay_length)   


    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+out_gap #Sequence duration

    seed_A = 1*np.array([[0],[1],[0],[1]])
    seed_B = 1*np.array([[0],[0],[1],[1]])

    #AND:
    and_y            =1* np.array([0,0,0,1])

    #OR:
    or_y            = np.array([0,1,1,1])

    #Context
    cont= np.array([[-1],[1]])

    '''
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap
    '''
    win              = signal.hann(10)
    
    out_t       = mem_gap+ first_in+stim_dur
    trial_types = np.random.randint(4, size=sample_size)
    trial_types_ = np.random.randint(2, size=sample_size)
    x_train     = np.zeros((sample_size, seq_dur, 3))
    x_train_       = np.zeros((sample_size, seq_dur, 3)) 
    y_train     =  np.zeros((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):
        seed()
        var_delay_length = 0#int(20*np.random.rand())
        var_high=0#0.25*np.random.rand()
        x_train[ii, first_in_c-var_delay_length:first_in_c + stim_dur, 2] =var_high+ cont[trial_types_[ii], 0]#context signal        
        x_train[ii, first_in-var_delay_length:first_in + stim_dur, 0] =var_high+ seed_A[trial_types[ii], 0]#input A
        x_train[ii, first_in-var_delay_length:first_in + stim_dur, 1] =var_high+ seed_B[trial_types[ii], 0]#input B

        if cont[trial_types_[ii], 0]==1: 
            y_train[ii, out_t:, 0] = and_y[trial_types[ii]]
        if cont[trial_types_[ii], 0]==-1: 
            y_train[ii, out_t:, 0] = or_y[trial_types[ii]]


    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
       
    x_train = x_train#+stim_noise * np.random.randn(sample_size, seq_dur, 3)##+np.ones((sample_size, seq_dur, 2) )#
    print("--- %s seconds to generate And dataset---" % (time.time() - start_time))
    return (x_train, y_train, mask,seq_dur)


#To see how is the training data set uncoment these lines

sample_size=10
mem_gap=20
x_train,y_train, mask,seq_dur = generate_trials(sample_size,mem_gap) 

#print ("x",x_train)
#print ("y",y_train)

fig     = plt.figure(figsize=(8,6))
fig.suptitle("\"And-Or\" Data Set Training Sample\n (amplitude in arb. units time in mSec)",fontsize = 20)
for ii in np.arange(10):
    
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 2],color='r',label="Context: \n1=And \n -1=OR")
    plt.plot(x_train[ii, :, 0],color='g',label="input A")
    plt.plot(x_train[ii, :, 1],color='b',label="input B")
    plt.plot(y_train[ii, :, 0],color='k',label="output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=1)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    figname = "plots/data_set_and_sample.png"
    #figname = base+"/data_set_and_sample.png" 
    plt.savefig(figname,dpi=200)
plt.show()


