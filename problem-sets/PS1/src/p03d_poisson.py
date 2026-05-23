import numpy as np
import util

from linear_model import LinearModel
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf=PoissonRegression(step_size=lr,max_iter=3000)
    clf.fit(x_train,y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    pred=clf.predict(x_eval)
    np.savetxt(pred_path,pred)

    loss=np.mean((pred-y_eval)**2)
    print(loss)

    plt.plot(y_eval, 'go', label='label')
    plt.plot(pred, 'rx', label='prediction')
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        #先初始化theta
        n=x.shape[1]
        m=y.shape[0]
        self.theta=np.zeros(n)
        
        for i in range(self.max_iter):
            h=np.exp(x@self.theta)#m

            update=self.step_size*(x.T@(y-h))/m

            if np.linalg.norm(update,ord=1)<self.eps:
                print(f"更新值小于{self.eps},停止更新")
                break

            self.theta+=update
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        h=np.exp(x@self.theta)
        return h
        # *** END CODE HERE ***
