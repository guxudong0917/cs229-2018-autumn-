import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels

    
    x_train,y_train=util.load_dataset(train_path,label_col='t',add_intercept=True)
    x_test,y_test=util.load_dataset(test_path,'t',add_intercept=True)

    clf=LogisticRegression()
    clf.fit(x_train,y_train)

    _,pred_c=clf.predict(x_test)
    np.savetxt(pred_path_c,pred_c)

    util.plot(x_test,y_test,clf.theta,pred_path.split(".txt")[0]+"testc")

    acc=np.mean(pred_c==y_test)
    print(acc)
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels

    x_train,y_train=util.load_dataset(train_path,label_col='y',add_intercept=True)
    x_test,y_test=util.load_dataset(test_path,'t',add_intercept=True)#测评时看真是标签

    clf.fit(x_train,y_train)

    _,pred_d=clf.predict(x_test)
    np.savetxt(pred_path_d,pred_d)

    util.plot(x_train,y_train,clf.theta,pred_path.split(".txt")[0]+"traind")
    util.plot(x_test,y_test,clf.theta,pred_path.split(".txt")[0]+"testd")

    acc=np.mean(pred_d==y_test)
    print(f"用t作为标签训练的正确率:{acc}")

    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid,y_valid=util.load_dataset(valid_path,add_intercept=True)

    h,_=clf.predict(x_valid[y_valid==1])#只用V+的做预测
    a=np.mean(h)

    h_test,_=clf.predict(x_test)
    pred_result=((h_test/a)>=0.5).astype(int)
    acc_fix=np.mean(pred_result==y_test)
    print(f"用测试集修正后的训练正确率:{acc_fix}")
    theta_a=clf.theta.copy()
    theta_a[0]=theta_a[0]-np.log(a/(2-a))
    util.plot(x_test,y_test,theta_a,pred_path.split(".txt")[0]+"etest_fix")
    np.savetxt(pred_path_e,pred_result)
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
