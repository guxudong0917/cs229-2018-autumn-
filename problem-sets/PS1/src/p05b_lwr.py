import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf=LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train,y_train)
    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    pred=clf.predict(x_eval)
    # Plot validation predictions on top of training set
    # plt.scatter(x_train,y_train,c="blue",marker="x")
    # plt.show()
    mse=np.mean((pred-y_eval)**2)
    print(f"测试集mse:{mse}")
    # No need to save predictions
    # Plot data
    # print(x_train.shape)

    train_pred=clf.predict(x_train)
    
    fig,axes=plt.subplots(1,2,figsize=(10,5))

    train_sort_idx = np.argsort(x_train[:, 1])
    axes[0].plot(x_train[train_sort_idx, 1], y_train[train_sort_idx], 'x',c="blue", label='train')
    axes[0].plot(x_train[train_sort_idx, 1], train_pred[train_sort_idx], 'o',c="red", label='train_pred')
    
    sort_idx = np.argsort(x_eval[:, 1])
    
    axes[1].plot(x_eval[sort_idx, 1], y_eval[sort_idx], 'x', c="blue",label='eval')
    axes[1].plot(x_eval[sort_idx, 1], pred[sort_idx], 'o', c="red",label='eval_pred')
    plt.legend()
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        m=x.shape[0]
        pred=[]
        for i in range(m):
            given_x=x[i].reshape(1,-1)#变为1,n 这样就可以广播

            w = np.exp(-(np.linalg.norm(self.x-given_x,ord=2,axis=1)**2)/(2*self.tau**2))

            W=np.diag(w)/2

            theta=np.linalg.solve(self.x.T@W@self.x,self.x.T@W@self.y)

            temp_pred=theta.T@x[i]
            pred+=[temp_pred]
        result=np.array(pred)
        return result
        # *** END CODE HERE ***
