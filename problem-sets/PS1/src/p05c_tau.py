import matplotlib.pyplot as plt
import numpy as np
import util

import pandas as pd

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    sort_idx=np.argsort(x_valid[:,1])

    plt.plot(x_valid[sort_idx,1],y_valid[sort_idx],label="valid",marker="x")
    mse_list=[]
    for tau in tau_values:
        clf=LocallyWeightedLinearRegression(tau=tau)
        clf.fit(x_train,y_train)

        pred=clf.predict(x_valid)

        plt.plot(x_valid[sort_idx,1],pred[sort_idx],label=f"valid_{tau}",marker="o")

        mse=np.mean((pred-y_valid)**2)
        mse_list+=[mse]

    plt.legend()
    plt.show()

    result=pd.DataFrame({"tau":tau_values,
                         "mse":mse_list}).sort_values(by="mse")
    print(result)
    best_tau=result.loc[1,"tau"]
    # Fit a LWR model with the best tau value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_clf=LocallyWeightedLinearRegression(tau=best_tau)
    best_clf.fit(x_train,y_train)
   
    # Run on the test set to get the MSE value
    new_pred=best_clf.predict(x_test)
    # Save predictions to pred_path
    np.savetxt(pred_path,new_pred)

    best_mse=np.mean((new_pred-y_test)**2)
    print(f"test mse:{best_mse}")
    # Plot data
    
    b_sort_idx=np.argsort(x_test[:,1])

    plt.plot(x_test[b_sort_idx,1],y_test[b_sort_idx],label="test",marker="x")
    plt.plot(x_test[b_sort_idx,1],new_pred[b_sort_idx],label="best_tau_pred",marker="o")

    plt.legend()
    plt.show()
    # *** END CODE HERE ***
