import numpy as np
import math

s = [1, -1]
b = [-2, -0.5, 0.5, 2]
X = np.array([[1,-1,-1,1], [1,1,-1,-1]]).reshape(2, 4)
Y = [1, -1, 1, -1]

def h(s, b, ele):
    if ele > b:
        a = s
    else:
        a = -s
    return a

def ft(w):
    minsum = 999999.9
    summ = 0.0
    #s
    for j in range(2):
        #b
        for k in range(4):
            #d
            for m in range(2):

                for i in range(4):
                    hh = h(s[j], b[k], X[m, i])
                    if not Y[i] == hh:
                        deter = 1
                    else:
                        deter = 0
                    #print(w[i])
                    summ += w[i] * deter
                #print(summ)
                if summ < minsum:
                    print("reach!!!!!!")
                    minsum = summ
                    mins = s[j]
                    minb = b[k]
                    mind = m

                summ = 0.0
                    # print(minsum)
    print(mins, minb, mind)
    return mins, minb, mind

#3.1
weight = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
ft(weight)


F = [1]*3

print(F[0]/sum(F))