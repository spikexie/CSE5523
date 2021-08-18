import numpy as np
import math

s = [1, -1]
b = [-2, -0.5, 0.5, 2]
#X = np.array([[1, -1, -1, 1], [1, 1, -1, -1]]).reshape(2, 4)
X = np.array([[0, -math.sqrt(2), 0, math.sqrt(2)], [math.sqrt(2), 0, -math.sqrt(2), 0]]).reshape(2, 4)
Y = [1, -1, 1, -1]

def adaboost_train(X, Y, max_iterations=4):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        max_iterations: the number of decision stumps to find
    Output:
        F: a data structure that records the max_iterations of decision stumps that you find
    Useful tool (you may or may not need them):
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """

    D = X.shape[0]  # feature dimension
    N = X.shape[1]  # number of data instances
    F = np.zeros([4, max_iterations])  # s,b,d three dimensions in order
    B = []
    S = [-1, 1]

    w = [1 / N] * N
    b = [-2, -0.5, 0.5, 2]
    b1 = [-2, -0.5, 0.5, 2]
    B.append(b)
    B.append(b1)
    ### Your job Q1 starts here ###

    for t in range(max_iterations):
        # step (a)
        minTotal = 9999999
        # s
        for s in range(2):
            # d
            for d in range(D):
                # b
                stump = B[d]
                for b in range(len(stump)):
                    # number of instance
                    total = 0.0
                    for i in range(N):
                        if X[d, i] > stump[b]:
                            hh = S[s]
                        else:
                            hh = -S[s]
                        if not Y[i] == hh:
                            deter = 1
                        else:
                            deter = 0
                        total += w[i] * deter
                    if total < minTotal:
                        minTotal = total
                        F[0][t] = S[s]
                        F[1][t] = stump[b]
                        F[2][t] = d
        # step (b)
        loss = 0.0
        for i in range(N):
            ele = F[2][t]
            ele = int(ele)
            if X[ele, i] > F[1][t]:
                hh = F[0][t]
            else:
                hh = -F[0][t]
            if not Y[i] == hh:
                deter = 1
            else:
                deter = 0
            loss += w[i] * deter
        print(t,": loss:", loss)
        # step (c)
        beta = 1 / 2 * math.log(((1 - loss) / loss))
        F[3][t] = beta
        # step (d)
        for i in range(N):
            ele = F[2][t]
            ele = int(ele)
            if X[ele, i] > F[1][t]:
                hh = F[0][t]
            else:
                hh = -F[0][t]
            if Y[i] == hh:
                w[i] = w[i] * math.exp(-beta)
            else:
                w[i] = w[i] * math.exp(beta)
            # for i in range(4):
            #     w[i] = w[i] / sum(w)
        w = w / np.sum(w)
        print(t, ": w: ", w)
    ### Your job Q1 ends here ###

    return F


def adaboost_accuracy(X, Y, F):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        F: a data structure that records the max_iterations of decision stumps that you find
    Output:
        accuracy: a scalar between 0 and 1
    """
    D = X.shape[0]  # feature dimension
    N = X.shape[1]  # number of data instances
    max_iterations = F.shape[1]
    ### Your job Q2 starts here ###
    correct = 0
    for i in range(N):
        total = 0.0
        sign = 0
        for t in range(max_iterations):
            ele = F[2][t]
            ele = int(ele)
            if X[ele, i] > F[1][t]:
                hh = F[0][t]
            else:
                hh = -F[0][t]
            total += hh * F[3][t]
        print(i+1,":", total)
        if total > 0:
            sign = 1
        elif total < 0:
            sign = -1

        if Y[i] == sign:
            correct += 1

    accuracy = correct * 1.0 / N
    ### Your job Q2 ends here ###

    return accuracy

# 3.1
weight = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
para = adaboost_train(X, Y, 3)
print("parameter:")
print(para)

adaboost_accuracy(X, Y, para)
