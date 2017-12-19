"""Optimizing parameters and hyperparameters for two layer convnet"""

#from tf_aerial_2_layers import *
import numpy as np
import matplotlib.pyplot as plt
from tf_optimize_2layers import *


def optimize_kernel_and_patch(subset_size, k_fold = 5):
    """Splits subset of data into k_fold number and cross-validates for kernel and patch size"""
    strides = 1
    depth_1st_layer = 32 #Keep power of 2
    kernel_sizes = [3,4,5,6,7]
    patch_sizes = [4,8,16]
    num_epochs = 5
    for k in range(len(patch_sizes)):
        print ("Patch size is", patch_sizes[k])
        patch_size = patch_sizes[k]
        accuracy = []
        stds = []
        for j in kernel_sizes:
            print ("Kernel size is", j)
            kernel_size = j
            errs= []
            for i in range(k_fold):
                print ("Current k-fold index is", i+1)
                e = main(subset_size,kernel_size,strides,depth_1st_layer, patch_size, num_epochs, 0.0, 0.01, i + 1, k_fold)
                print ("This error -", e, "- will be used in cross validation")
                errs.append(e)
            validation_accuracy = 100 - np.mean(errs)
            std_dev = np.std(errs)
            accuracy.append(validation_accuracy)
            stds.append(std_dev)
            
        plt.figure(1)
        #plt.plot(kernel_sizes,accuracy, label = "Patch size = {}".format(patch_size),color='C{}'.format(k))
        plt.errorbar(kernel_sizes, accuracy, xerr=0,yerr=stds,label="Patch size={}".format(patch_size),mew = 2)
        print (stds, "The error bars")
        plt.xlabel("Kernel Size, F")
        plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return accuracy

def optimize_num_epochs(subset_size, k_fold = 5):
    """Splits subset of data into k_fold number and cross-validates for number of epochs"""
    strides = 1
    depth_1st_layer = [16,32,64] #Keep power of 2
    kernel_size = 3 #Choose "best" one
    patch_size = 8 #Choose "best" one
    num_epochs = np.linspace(5,20,4)#Careful to ensure epochs are integers
    for k in depth_1st_layer:
        print ("Depth of 1st layer is", k)
        accuracy = []
        stds = []
        for i in num_epochs:
            print ("Number of epochs is", i)
            errs = []
            for j in range(k_fold):
                print ("Current k-fold index is", j+1)
                e = main(subset_size,kernel_size,strides,k, patch_size, i, 0.0, 0.01, j + 1, k_fold)
                print ("This error -", e, "- will be used in cross validation")
                errs.append(e)
            validation_accuracy = 100 - np.mean(errs)
            std_dev = np.std(errs)
            accuracy.append(validation_accuracy)
            stds.append(std_dev)
        plt.figure(2)
        #plt.plot(num_epochs,accuracy,label = )
        plt.errorbar(num_epochs, accuracy, xerr=0, yerr=stds,label="Depth 1st Layer = {}".format(k),mew = 2)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return accuracy

def optimize_hyperparam(subset_size, k_fold =5):
    """Uses k-fold corss-valdation to optimize learning rate and momentum. Use only for deeper nets?"""
    strides = 1
    depth_1st_layer = 64 #Keep power of 2
    kernel_size = 3 #Choose "best" one
    patch_size = 4 #Choose "best" one
    num_epochs = 5
    momenta = [0.0,0.5,0.8,1.2]
    rates = np.linspace(0.0001,0.1,10)
    for i in momenta:
        print ("Momentum is", i)
        accuracy = []
        stds = []
        for k in rates:
            print ("Learning rate is", k)
            errs = []
            for j in range(k_fold):
                print ("Current k-fold index is", j+1)
                e = main(subset_size,kernel_size,strides,depth_1st_layer, patch_size, num_epochs, i, k, j + 1, k_fold)
                print ("This error -", e, "- will be used in cross validation")
                errs.append(e)
            validation_accuracy = 100 - np.mean(errs)
            std_dev = np.std(errs)
            accuracy.append(validation_accuracy)
            stds.append(std_dev)
        plt.figure(3)
        #plt.plot(rates,accuracy, label = "Momentum = {}".format(i))
        plt.errorbar(rates, accuracy, xerr=0, yerr=stds,label="Momentum = {}".format(i),mew = 2)
        plt.xlabel("Learning rates")
        plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return accuracy    
        
def test(kernel_size):
    """Just to check whether I need to "feed value" after iterations"""
    subset_size = 5
    strides = 1
    depth_1st_layer = 32
    for i in range(3):
        main(subset_size,kernel_size,strides,depth_1st_layer)
    return None