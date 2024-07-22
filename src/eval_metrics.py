import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)


# def eval_mosi(results, truths, exclude_zero=False):
#     return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, epoch):
    emos = ["硬度", "粗糙度"]
    test_preds = results.view(-1, 2, 4).cpu().detach().numpy()
    test_truth = truths.view(-1, 2).cpu().detach().numpy()

    for i in range(test_truth.shape[0]):
        temp = test_truth[i, 0]
        test_preds[i, 0, temp] = test_preds[i, 0, temp]
        temp2 = test_truth[i, 1]
        test_preds[i, 1, temp2] = test_preds[i, 1, temp2]

    for emo_ind in range(2):
        print(f"{emos[emo_ind]}: ")
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        diff_in = np.where(test_truth_i != test_preds_i)[0]
        print('different is', test_preds_i[diff_in])
        print('truth is ', test_truth_i [diff_in])
        print("  - f1score:", f1)
        print("  - Accuracy: ", acc)

def eval_iemocap_acc(results, truths, epoch):

    test_preds = results.view(-1, 2, 4).cpu().detach().numpy()
    test_truth = truths.view(-1, 2).cpu().detach().numpy()
    acc_two = np.zeros([2])
    f1_two = np.zeros([2])
    total = np.ones([test_truth.shape[0]])
    # print(total.shape)
    total_preds = np.zeros([test_truth.shape[0]])
    for i in range(test_truth.shape[0]):
        temp = test_truth[i, 0]
        test_preds[i, 0, temp] = test_preds[i, 0, temp]
        temp2 = test_truth[i, 1]
        test_preds[i, 1, temp2] = test_preds[i, 1, temp2]

    preds_temp = np.zeros([test_truth.shape[0], 2])
    for emo_ind in range(2):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        # print(test_preds_i.shape)
        preds_temp[:, emo_ind] = test_preds_i
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        f1_two[emo_ind] = f1
        acc_two[emo_ind] = acc
    for i in range(test_truth.shape[0]):
        if preds_temp[i, 0] == test_truth[i, 0] and preds_temp[i, 1] == test_truth[i, 1]:
            total_preds[i] = 1

    acc_total = accuracy_score(total, total_preds)
    f1_total = f1_score(total, total_preds, average='weighted')

    return acc_two[0], acc_two[1], acc_total, f1_two[0], f1_two[1], f1_total


