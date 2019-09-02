import platform
import os
import getpass
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
from MetricLearning.AdaptiveMetricModel import AdaptiveMetricModel


def load_model(path):
    model = AdaptiveMetricModel.load(path)
    return model


def similarity_score(x1, x2, model, threshold, n_layer):
    dists = {}

    for layer_idx in range(n_layer):
        x1_prime = model.transform(x1, layer_idx=layer_idx)
        x2_prime = model.transform(x2, layer_idx=layer_idx)
        dist = np.linalg.norm(x1_prime - x2_prime)
        dist /= 2
        dists[layer_idx] = dist

    p = 0
    for layer_idx in dists:
        dist = dists[layer_idx]
        if dist < threshold:
            p += 1 * model.alpha[layer_idx]

    return p


def compute_roc(fold, model, data, thresholds, n_layer):
    fprs = []
    tprs = []

    accs = []
    for threshold in thresholds:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(data)):
            x1, x2, l = data[i]
            p = similarity_score(x1, x2, model, threshold, n_layer)

            if l == 1:
                if p > 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if p > 0.5:
                    FP += 1
                else:
                    TN += 1

            if p > 0.5:
                predict = 1
            else:
                predict = -1
            print("Fold -- {} -- Threshold {} -- Test case {} -- p {} -- predict {} -- true {}".format(
                fold, threshold, i, p, predict, l))
        fprs.append(FP/(FP+TN))
        tprs.append(TP/(TP+FN))
        accs.append((TP+TN)/(TP+TN+FP+FN))
    best_accuracy = np.max(accs)
    print("Best Accuracy: ", best_accuracy)
    return fprs, tprs, best_accuracy


def compare_roc(names, fpr, tpr):
    fig, ax = plt.subplots()
    for each in names:
        f = fpr[each]
        t = tpr[each]
        auc_ = auc(f, t)
        ax.plot(f, t, '--', markersize=5, label='{} ({:.3f})'.format(each, auc_), linewidth=3.5)
    plt.xlabel("FPR", fontsize=15, fontweight='bold')
    plt.ylabel("TPR", fontsize=15, fontweight='bold')
    legend = plt.legend(prop={'weight': 'bold', 'size': 15})
    # legend = plt.legend(fontsize=13)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("black")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Roc Curves for LFW", fontsize=15, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize=15)
    plt.setp(ax.get_xticklabels(), fontweight="bold", fontsize=15)
    plt.show()


def plot_alpha(path, segment=-1):
    data = pd.read_csv(path)
    alpha = data['Alpha']

    alpha = alpha.values
    for i, each in enumerate(alpha):
        tmp = each.split(',')
        tmp = [float(each) for each in tmp]
        alpha[i] = tmp
    alpha = np.array(alpha)

    if segment == -1:
        # plot final weights
        value = alpha[-1]
        fig, ax = plt.subplots()
        x = np.arange(len(value))
        names = [r'$\mathbf{E0}$', r'$\mathbf{E1}$', r'$\mathbf{E2}$', r'$\mathbf{E3}$', r'$\mathbf{E4}$',
                 r'$\mathbf{E5}$']
        colors = ['black', 'green', 'blue', 'red', '#db4429', '#29d2db']
        width = 0.7
        opacity = 0.7
        for i in range(len(x)):
            ax.bar(x[i], value[i], width=width, color=colors[i], ecolor='#c109ea', alpha=opacity)
        ax.set_ylabel(r"Metric Model Weight (0-1)", size=15, fontweight='bold')
        ax.set_xlabel('Metric Models', size=15, fontweight='bold')
        plt.xticks(x, names, fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=15, fontweight="bold")
        plt.setp(ax.get_xticklabels(), fontsize=15, fontweight="bold")
        plt.title('Weight Distribution of Metric Models', fontsize=15, fontweight='bold')
        plt.show()


if __name__ == '__main__':
    os_name = platform.system()
    data_path_ = 'demo/lfw_attribute.csv'
    model_path_ = 'demo/final.pth'
    meta_path = 'demo/lfw_attribute_test.bin'

    """ Load Model and create plot object """
    print("Start Loading Model", flush=True)
    model_ = load_model(model_path_)
    print("Model Loading Done!", flush=True)

    """ Compute ROC for Ours (LFW) """
    with open(meta_path, 'rb') as f:
        test_data_ = pickle.load(f)
    tmp_tpr = []
    tmp_fpr = []
    tmp_acc = []

    for fold in test_data_:
        print("Fold: ", fold)
        our_fpr, our_tpr, acc = compute_roc(fold, model_, test_data_[fold], np.arange(0, 1, 0.01), n_layer=6)
        tmp_fpr.append(our_fpr)
        tmp_tpr.append(our_tpr)
        tmp_acc.append(acc)
    our_fpr = np.mean(tmp_fpr, axis=0)
    our_tpr = np.mean(tmp_tpr, axis=0)
    average_best_acc = np.mean(tmp_acc, axis=0)
    print("Average Best Accuracy: ", average_best_acc, np.std(tmp_acc))

    # with open('ours_feature_lfw.bin', 'wb') as f:
    #     pickle.dump({'tprs': our_tpr, 'fprs': our_fpr}, f)

    """ Plot ROC figure for Attribute Features """
    our_name = 'OAHU'
    username = getpass.getuser()
    with open('demo/ours_attribute_lfw.bin', 'rb') as f:
        tmp = pickle.load(f)
    our_tpr = tmp['tprs']
    our_fpr = tmp['fprs']

    with open('demo/oasis_attribute_lfw.bin', 'rb') as f:
        tmp = pickle.load(f)
    oasis_tpr = tmp['tprs']
    oasis_fpr = tmp['fprs']
    with open('demo/opml_attribute_lfw.bin', 'rb') as f:
        tmp = pickle.load(f)
    opml_tpr = tmp['tprs']
    opml_fpr = tmp['fprs']
    with open('demo/rdml_attribute_lfw.bin', 'rb') as f:
        tmp = pickle.load(f)
    rdml_tpr = tmp['tprs']
    rdml_fpr = tmp['fprs']

    with open('demo/lego_attribute_lfw.bin', 'rb') as f:
        tmp = pickle.load(f)
    lego_tpr = tmp['tprs']
    lego_fpr = tmp['fprs']
    fpr = {
        our_name: our_fpr,
        'OASIS': oasis_fpr,
        'OPML': opml_fpr,
        'RDML': rdml_fpr,
        'LEGO': lego_fpr
    }
    tpr = {
        our_name: our_tpr,
        'OASIS': oasis_tpr,
        'OPML': opml_tpr,
        'RDML': rdml_tpr,
        'LEGO': lego_tpr
    }

    oasis_fpr = np.array(oasis_fpr)
    oasis_tpr = np.array(oasis_tpr)
    our_fpr = np.array(our_fpr)
    our_tpr = np.array(our_tpr)
    opml_fpr = np.array(opml_fpr)
    opml_tpr = np.array(opml_tpr)

    tmp_fpr = sorted([(i, each) for i, each in enumerate(oasis_fpr)], key=lambda x: x[1])
    tmp_idx = [each[0] for each in tmp_fpr]
    oasis_fpr = oasis_fpr[tmp_idx]
    oasis_tpr = oasis_tpr[tmp_idx]

    tmp_fpr = sorted([(i, each) for i, each in enumerate(our_fpr)], key=lambda x: x[1])
    tmp_idx = [each[0] for each in tmp_fpr]
    our_fpr = our_fpr[tmp_idx]
    our_tpr = our_tpr[tmp_idx]

    tmp_fpr = sorted([(i, each) for i, each in enumerate(opml_fpr)], key=lambda x: x[1])
    tmp_idx = [each[0] for each in tmp_fpr]
    opml_fpr = opml_fpr[tmp_idx]
    opml_tpr = opml_tpr[tmp_idx]

    print("AUC of OASIS: ", auc(oasis_fpr, oasis_tpr))
    print("AUC of OPML: ", auc(opml_fpr, opml_tpr))
    print("AUC of RDML: ", auc(rdml_fpr, rdml_tpr))
    print("AUC of LEGO: ", auc(lego_fpr, lego_tpr))
    print("AUC of ours: ", auc(our_fpr, our_tpr))

    oasis_auc = auc(oasis_fpr, oasis_tpr)
    opml_auc = auc(opml_fpr, opml_tpr)
    rdml_auc = auc(rdml_fpr, rdml_tpr)
    lego_auc = auc(lego_fpr, lego_tpr)
    ours_auc = auc(our_fpr, our_tpr)

    tmp = [(our_name, ours_auc), ('OASIS', oasis_auc), ('OPML', opml_auc), ('RDML', rdml_auc), ('LEGO', lego_auc)]
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    names = [each[0] for each in tmp]
    compare_roc(names, fpr, tpr)



