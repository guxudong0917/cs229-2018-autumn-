import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    #训练时不要开启intercept,即加上x0=1(b),这样x0的方差全为0，会导致协方差矩阵不可逆
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # x_train=np.stack((x_train[:,0],np.log(x_train[:,1])),axis=1)

    # *** START CODE HERE ***


    clf=GDA()
    clf.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # x_eval=np.stack((x_eval[:,0],x_eval[:,1],np.log(x_eval[:,2])),axis=1)
    
    _,pred=clf.predict(x_eval)
    acc=np.mean(pred==y_eval)
    print(f"GDA正确率{acc}")

    train_path=pred_path.split(".txt")[0]+"train_trans"
    eval_path=pred_path.split(".txt")[0]+"eval_trans"
    util.plot(x_train,y_train,clf.theta,train_path)
    util.plot(x_eval,y_eval,clf.theta,eval_path)
    np.savetxt(pred_path,pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        fai=np.mean(y==1)

        #如果要用*的话，得先变形成(m,1)
        # y_0=(y==0).reshape(-1,1)
        # y_1=(y==1).reshape(-1,1)

        u0=np.sum(x[y==0],axis=0)/np.sum(y==0)
        u1=np.sum(x[y==1],axis=0)/np.sum(y==1)

        m=y.shape[0]
        n=x.shape[1]
        sig=np.zeros((n,n))

        for i in range(m):
            if y[i]==0:
                u_y=u0
            else:
                u_y=u1

            z_i=(x[i]-u_y).reshape(-1,1)
            sig=sig+z_i@z_i.T

        sig=sig/m

        sig_neg=np.linalg.inv(sig)
        theta=sig_neg@((u1-u0).reshape(-1,1))#(n,1)
        theta_0=((u0-u1).reshape(-1,1)).T@sig_neg@((u1+u0).reshape(-1,1))/2-np.log((1-fai)/fai)
        #(1,1)

        self.theta=np.concatenate([[theta_0.item()],theta.flatten()])

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        f=x@self.theta

        pred=1/(1+np.exp(-f))

        return pred,(pred>=0.5).astype(int)
        # *** END CODE HERE
