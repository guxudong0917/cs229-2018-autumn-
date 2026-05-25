# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0
    while True:
        i += 1
        prev_theta = theta.copy()
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * (grad+0*theta)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            delta = np.linalg.norm(prev_theta - theta)
            print(np.linalg.norm(theta))
            print(delta)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)

    
    # logistic_regression(Xa, Ya)


    # plt.scatter(Xa[Ya==1,-2],Xa[Ya==1,-1],marker="o",c="r",label="1")
    # plt.scatter(Xa[Ya==-1,-2],Xa[Ya==-1,-1],marker="x",c="b",label="-1")
    # plt.legend()
    # plt.show()
    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    m=Xb.shape[0]
    n=Xb.shape[1]

    Xb+=0.1*np.random.randn(m,n)
    plt.scatter(Xb[Yb==1,-2],Xb[Yb==1,-1],marker="o",c="r",label="1")
    plt.scatter(Xb[Yb==-1,-2],Xb[Yb==-1,-1],marker="x",c="b",label="-1")
    plt.legend()
    plt.show()
    
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
