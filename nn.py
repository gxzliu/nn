"""
nn.py
george liu
date: 4.16

functions to return the most likely labels of the test data 
and to determine how accurate these are.

"""


import numpy as np
from scipy import spatial
    
    
def nnclass(training ,test):   
    '''
    Classifier function that takes in labeled training data and nonlabled
    test data and uses nearest neighbor to  returns the most
    likely test label.
    ''' 
    
    ordered = []
    for r in range(len(test)):    
        valcount = None
        distcount = 0
        for t in range(len(training)):
            #fills an array with a row from test
            tester = test[r]
            #fills an array with a row of of training but w/o the label
            trainer = training[t][1:]
            euclid = spatial.distance.euclidean(tester, trainer)
            #finds the smallest distance and gets the location value
            if (distcount == 0 or euclid < distcount):
                distcount = euclid
                valcount = training[t][0]
        ordered.append(valcount)
    ans = np.asarray(ordered)
    return ans
    

def nvalid(data, p, classifier, *args):
    '''
    Uses p-fold validation on data using a given classifier, estimating
    the performance of the classifier.
    '''
    
    np.random.shuffle(data)
    #finds how many values are in each section
    partition = len(data) // p
    correct = []
    for r in range(p):
        #determines where the test data will start and stop 
        min = r * (partition)
        max  = (partition) * (r + 1)
        divider = range(min, max)
        #makes the test data from overall data
        tester = data[divider]
        #makes training data by deleting the testing data from overall data
        trainer = np.delete(data, divider, 0)
        actual = tester[:,0]
        #deletes column so same size as training
        nntest = np.delete(tester, 0, 1)
        #use classifier to get the predicted values
        expected = classifier(trainer, nntest)
        correct.append(compare(data, expected, actual))
    return sum(correct)
            

def compare(data, simulate, real):
    '''
    Compares the simulated results to actual results
    '''
    #match is a counter for how many times there is a match
    matches = 0
    for j in range(simulate.shape[0]):
        if simulate[j] == real[j]:
            matches = matches + 1
    #returns percentage correct
    return matches/data.shape[0]
