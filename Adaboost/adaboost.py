from numpy import matrix

def loadSimpData():
    data_mat = matrix([[1.  ,  2.1],
        [2.  ,  1.1],
        [1.3 ,  1. ],
        [1.  ,  1. ],
        [2.  , 1.  ]])
    class_label = [1.0, 1.0, -1.0, -1.0, 1.0]

    return data_mat,class_label