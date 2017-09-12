# basic script to load the data

import numpy as np
import os


def load_data(path="/usr/local/diet_code/genomic", norm=True):
    """
    Load the train, valid, test data.
    Args:
        - path: path to the main dir containing {trainX.npy, validX.npy, ...}
                assumes they are named that way...
    Returns:
        - trainX, trainY, validX, validY, testX, testY: the data!!
    """
    
    print("here")
    if norm:
        if os.path.isfile(os.path.join(path,"ntrainX.npy")): # the standardized data exists
            trainX = np.load(os.path.join(path,"ntrainX.npy"))
            trainY = np.load(os.path.join(path,"trainY.npy"))
            validX = np.load(os.path.join(path,"nvalidX.npy"))
            validY = np.load(os.path.join(path,"validY.npy"))
            testX = np.load(os.path.join(path,"ntestX.npy"))
            testY = np.load(os.path.join(path,"testY.npy"))
            
        else: # standardize the data
            trainX = np.load(os.path.join(path,"trainX.npy"))
            trainY = np.load(os.path.join(path,"trainY.npy"))
            validX = np.load(os.path.join(path,"validX.npy"))
            validY = np.load(os.path.join(path,"validY.npy"))
            testX = np.load(os.path.join(path,"testX.npy"))
            testY = np.load(os.path.join(path,"testY.npy"))
            
            # standardize the data
            print("standardizing data")
            X = np.concatenate([trainX,validX,testX])
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            trainX = (trainX - mu[None,:]) / sigma[None,:]
            validX = (validX - mu[None,:]) / sigma[None,:]
            testX = (testX - mu[None,:]) / sigma[None,:]

            #save the standardized data:
            np.save(os.path.join(path, "ntrainX.npy"), trainX)
            np.save(os.path.join(path, "ntestX.npy"), testX)
            np.save(os.path.join(path, "nvalidX.npy"), validX)
            print("saved and standardized.")

    else:
        trainX = np.load(os.path.join(path,"trainX.npy"))
        trainY = np.load(os.path.join(path,"trainY.npy"))
        validX = np.load(os.path.join(path,"validX.npy"))
        validY = np.load(os.path.join(path,"validY.npy"))
        testX = np.load(os.path.join(path,"testX.npy"))
        testY = np.load(os.path.join(path,"testY.npy"))

    return trainX, trainY, validX, validY, testX, testY


if __name__=="__main__":
    load_data()
