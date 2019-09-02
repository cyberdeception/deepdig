
#from Encoder import autoencode

import platform
import os
from Utility.util import Utility
from CmdHelp.CmdHelp import CmdHelper
from logger import Logger, OMLLogger
from oml import OML
import pandas as pd
import numpy as np
import torch
import getpass
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import os.path






def calcTPR_FPR(actual, predicted):
    positive = [22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
    negative = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for entry in zip(actual,predicted):
        if entry[0] in positive: # actual is positive
            if entry[1] in positive: # predicted is positive too
                tp += 1
            else: # predicted is negative
                fn += 1
        elif entry[0] in negative: # actual is negative
            if entry[1] in positive: # predicted is positive
                fp += 1
            else: # predicted is negative too
                tn += 1

    tpr = str( "recall TPR %.4f" % (float(tp)/float(tp+fn)) )
    fpr = str( "FPR %.4f" % (float(fp)/float(fp+tn) ))
    Acc = str( "Accuracy %.4f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F2  = str( "F2 %.4f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    #precision = tp / float(tp + fp)
    #print("precision " + str(precision))
    print(tpr)
    print(fpr)
    print(Acc)
    print(F2)


def readData(filepath,classes):
    #c = "".join(str(x) for x in classes)
    appendname = ".float"
    if os.path.exists(filepath+appendname):
        print(filepath+appendname + str(" exists"))
        data = pd.read_csv(filepath+appendname, header=0)
        data = select_class(data, classes)
        return data
    else:
        print(filepath+appendname + str(" does not exists create a cache"))
        data = pd.read_csv(filepath, header=0)
        data = select_class(data, classes)
        data = convertToFloat(data)
        data.to_csv(filepath+appendname,index=False)
        return data
        
def select_class(data, classes):
    if len(classes) == 0:
        return data
    data = data.loc[data['class'].isin(classes)]
    return data

def convertToFloat(df):
    cols = df.columns.drop('class')
    for col in cols:
        df[col] = df[col].astype(float)
    return df

le = preprocessing.LabelEncoder()
if __name__ == '__main__':
    os_name = platform.system()

    # path to dataset
    data_path = '/home/debo/Downloads/alldataw2vnoben.csv'
    #data_path = "/home/debo/Downloads/www2019/emnist.csv"
    # experiment configuration name
    data_name = 'cfg'


    args = Utility.check_args(CmdHelper.get_parser())
    print(args.trainpath)   
    """ preprocessing """
    username = getpass.getuser()
    data_path = os.path.join('/home/'+username, data_path)
    log_path = os.path.join(args.base_dir, args.log_dir, data_name)
    checkpoint_dir = os.path.join(args.base_dir, args.checkpoint_dir, data_name)
    args.result_dir = os.path.join(args.base_dir, args.result_dir, data_name)

    Utility.setup(args)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    metric_logger = Logger(log_path)

    log_path = os.path.join(log_path, 'oml_train.log')

    oml_logger = OMLLogger.getLogger(log_path)

    """read data and split"""
    datapath_train = args.trainpath #"/home/debo/Downloads/ngramnormaltrain.csv"
    datapath_test = args.testpath #"/home/debo/Downloads/ngramnormaltest.csv"
    c = args.classes
    classes = []
    if len(c) > 0:  
        for x in c.split(","):
            classes.append(int(x))
    datatrain = readData(datapath_train, classes)
    
    print(datatrain)
    classes = []
    datatest = readData(datapath_test, classes)
    

    print("Data Reading Done")
    #n, _ = data.shape
    from sklearn.model_selection import train_test_split

        # randomly split
    train_data =  datatrain.values #convertToFloat(datatrain).values
    valid_data = None
    test_data =  datatest.values #convertToFloat(datatest).values
    t, _ = test_data.shape
    t = t - 1
            # get features and labels
    train_feature = train_data[1:, :-1]
    #train_feature = train_data[1:,0:t]
    train_label = train_data[1:, -1]
    #train_label = np.asarray([0.0 if x < 18 else 1.0 for x in train_label])
    valid_feature = None
    valid_label = None
    test_feature = test_data[1:, :-1]
    #test_feature = test_data[1:, 0:t]
    test_label = test_data[1:, -1]
    print(len(test_label))
    #test_label = np.asarray([0.0 if x < 18 else 1.0 for x in test_label])
    print(len(test_label))
    rtio = 0.30

    #train_feature, _, train_label, _ = train_test_split(train_feature, train_label, train_size=rtio)
    #test_feature, _, test_label, _ = train_test_split(test_feature, test_label, train_size=rtio)
    #train_feature, test_feature = autoencode(train_feature, test_feature)

    """create OML object and start"""
    oml = OML(args, train_feature, train_label, metric_logger, oml_logger, checkpoint_dir)

    # start training
    oml.start(valid_feature=valid_feature, valid_label=valid_label, evaluate_valid=False)

    # build knn classifier
    oml.build_knn_classifier(train_feature, train_label, k=5, cuda=torch.cuda.is_available(), args=args)
    # predict on test dataset
    p_label, p_prob = oml.predict(test_feature)

    """analysis"""
    calcTPR_FPR(test_label, p_label)
    
    #acc = np.sum(p_label == test_label)/len(test_label)
    #print("Accuracy: ", acc, "Macro F1: ", f1_score(test_label, p_label, average='macro'))
    """confusion = confusion_matrix(test_label, p_label)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]


    false_positive_rate = FP / float(TN + FP)
    print("fpr " + str(false_positive_rate))

    precision = TP / float(TP + FP)
    print("precision " + str(precision))

    recall = TP / float(TP + FN)
    print("recall tpr " + str(recall))
    """ 

    """save prediction result"""
    result_data = np.concatenate((test_label.reshape(-1, 1), p_label.reshape(-1, 1), p_prob.reshape(-1, 1)), axis=1)
    np.savetxt(os.path.join(args.result_dir, args.result_file), result_data, delimiter=',')
   
