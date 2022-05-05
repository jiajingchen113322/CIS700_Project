import numpy as np

def get_acc(pred,target):
    pred=np.argmax(pred,-1)
    acc=np.sum(pred==target)/len(pred)
    return acc