import numpy as np
import math

s = [1, -1]
b = [-2, -0.5, 0.5, 2]
d = [0, 1]
# data
X = np.array([[0, -math.sqrt(2), 0, math.sqrt(2)], [math.sqrt(2), 0, -math.sqrt(2), 0]]).reshape(2, 4)
Y = [1, -1, 1, -1]


def h(s, b, ele):
    if ele > b:
        a = s
    else:
        a = -s
    return a


weight = [1 / 4, 1 / 4, 1 / 4, 1 / 4]


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
                print(summ)
                if summ < minsum:
                    print("reach!!!!!!")
                    #print(summ)
                    minsum = summ
                    mins = s[j]
                    print(mins)
                    minb = b[k]
                    print(minb)
                    mind = m
                    print(mind)
                summ = 0.0
                    # print(minsum)
    return mins, minb, mind


def loss(mins, minb, mind):
    summ = 0.0
    for i in range(4):
        hh = h(mins, minb, X[mind, i])
        if not Y[i] == hh:
            deter = 1
        else:
            deter = 0
        summ += 1 / 4 * deter
    return summ


def beta(lss):
    return 1 / 2 * math.log(((1 - lss) / lss))


def updateBeta(w, bta):
    for i in range(4):
        hh = h(mins, minb, X[mind, i])
        if not Y[i] == hh:
            w[i] = w[i] * math.exp(-bta)
        else:
            w[i] = w[i] * math.exp(bta)
    for i in range(4):
        w[i] = w[i] / sum(weight)

    return w


mins, minb, mind = ft(weight)
# print("mins, minb, mind, minsum")
print(mins, minb, mind)
loss1 = loss(mins, minb, mind)
print(loss1)
beta1 = beta(loss1)
print(beta1)

weight1 = updateBeta(weight, beta1)
print("weight")
print(weight1)
# t = 2
print("----t = 2----")
mins2, minb2, mind2 = ft(weight1)
# print("mins, minb, mind, minsum")
print(mins2, minb2, mind2)
loss2 = loss(mins2, minb2, mind2)
print(loss2)
beta2 = beta(loss2)
print(beta2)

weight2 = updateBeta(weight1, beta2)
print("----t = 3----")
mins3, minb3, mind3 = ft(weight2)
# print("mins, minb, mind, minsum")
print(mins3, minb3, mind3)
loss3 = loss(mins3, minb3, mind3)
print(loss3)
beta3 = beta(loss3)
print(beta3)