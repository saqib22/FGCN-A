import numpy as np
from Parameters import Parameters
from sklearn.preprocessing import label_binarize

def my_binarize(labels, classes):
    para = Parameters()
    if len(classes) == 0:
        classes = [i for i in range(para.outputClassN)]
    
    outputLabels = []

    for i, label in enumerate(labels):
        binarize = label_binarize(label, classes)
        outputLabels.append(binarize)
    
    return np.array(outputLabels)

if __name__ == "__main__":
    labels = np.random.randint(low=0, high=14,size=(1000, 4096))
    output = my_binarize(labels, [])

    print (output.shape)