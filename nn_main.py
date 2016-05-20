"""
nn_main.py
george liu
date: 4.16

functions to run nn.py on the formatted wdbc data in the folder and 
to generate synthetic data to compare accuracies between the synthetic
and real data.

"""

import nn
import numpy as np

def main():
    '''
    Creates synthetic data then runs the nvalidator on the synthetic and breast
    cancer data. Class 1: mean = [2.5, 3.5], covariance = [[1, 1], [1, 4.5] ], 
    300 samples. Class 2: mean = [.5, 1], covariance = [[2, 0], [0, 1]], 
    300 samples.
    '''
    
    #reads, cleans, and shapes data
    file = open("wdbc.data.txt", 'r')
    inputval = np.fromfile(file, sep=" ")
    inputval.shape = (569,32)
    real = np.delete(inputval,0,1)
    
    x1=2.5
    x2=0.5
    y1=3.5
    y2= 1
    mean_1 = [x1,y1]
    mean_2 = [x2,y2]
    cov_1=[[1,1],[1,4.5]]
    cov_2=[[2,0],[0,1]]
    n_2 = 300
    n_1 = 300
    
    synth = synthetic(mean_1, mean_2, cov_1, cov_2, n_1, n_2)
    
    #tests synthetic data with 5-fold cross-validation
    print('Classifier accuracy: ', nn.nvalid(synth, 5 , nn.nnclass))
    
    #tests real data with 5-fold cross-validation
    print('Classifier accuracy: ', nn.nvalid(real, 5 , nn.nnclass))
    
    
def synthetic(firstm, secm, firstc, secc, firstn, secn):
    '''
    Creates a synthetic data set from multivariate normal distributions
    '''
    
    set1= np.random.multivariate_normal(firstm,firstc,firstn)
    set2= np.random.multivariate_normal(secm,secc,secn)
    
    #creates the first synthetic label
    labels = np.zeros(firstn)
    labels.shape = (firstn , 1)
    #adds the label to data by stacking horiz
    set1_label = np.hstack((labels , set1))
    
    labels2 =  np.ones(secn)
    labels2.shape = (secn , 1)
    set2_label = np.hstack((labels2 , set2))
    
    #combines all data
    data = np.vstack((set1_label, set2_label))
    np.random.shuffle(data)
    
    return(data)

main()

