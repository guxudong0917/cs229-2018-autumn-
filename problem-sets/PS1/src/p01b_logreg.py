import numpy as np
import util

from linear_model import LinearModel

import os
def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    clf=LogisticRegression()
    clf.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    _,pred=clf.predict(x_eval)
    acc=np.mean(pred==y_eval)
    print(acc)

    train_path=pred_path.split(".txt")[0]+"train"
    eval_path=pred_path.split(".txt")[0]+"eval"
    util.plot(x_train,y_train,clf.theta,train_path)
    util.plot(x_eval,y_eval,clf.theta,eval_path)
    np.savetxt(pred_path,pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        #先初始化theta
        n=x.shape[1]
        m=y.shape[0]
        self.theta=np.zeros(n)
        # 这里是一次把所有的样本都放进去迭代
        for i in range(self.max_iter):
            f=x@self.theta #m,
            h=1/(1+np.exp(-f))

            J=-np.mean(y*np.log(h)+(1-y)*np.log(1-h))

            if self.verbose:
                print(f"loss:{J}")

            J_grad=(x.T@(h-y))/m
            Hessian=(x.T@np.diag(h*(1-h))@x)/m # (n,n)
            #更新是要求Hessian的逆与J_grad矩阵相乘，下面求解x，Hessian@x=J_grad，故满足
            update=self.step_size*np.linalg.solve(Hessian,J_grad) #(n,)

            if np.linalg.norm(update,ord=1)<self.eps:
                print(f"更新值小于{self.eps},停止更新")
                break
            self.theta-=update

        # *** END CODE HERE ***
        

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        f=x@self.theta #m,
        h=1/(1+np.exp(-f))

        return h,(h>=0.5).astype(int)
    
        # *** END CODE HERE ***
