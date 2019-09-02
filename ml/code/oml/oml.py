import numpy as np
import torch
import os
import csv
from MetricLearning.AdaptiveMetricModel import AdaptiveMetricModel
from sklearn.neighbors import KNeighborsClassifier
from Utility.GpuKnn import GpuKNN
import progressbar


class OML(object):
    def __init__(self, args, train_feature, train_label, metric_logger=None, oml_logger=None, checkpoint_dir=None,
                 preload=None):
        self.args = args
        self.metricLogger = metric_logger
        self.omlLogger = oml_logger
        self.checkpoint_dir = checkpoint_dir
        device = torch.device("cuda:"+args.gpu_id if args.cuda else "cpu")

        # create repository
        self.repository = {}

        for i, l in enumerate(train_label):
            if l not in self.repository:
                self.repository[l] = []
            self.repository[l].append(i)
        self.labelSet = list(self.repository.keys())

        # convert training data to tensors
        self.train_feature = torch.from_numpy(train_feature)
        self.train_label = train_label

        if self.args.cuda:
            self.train_feature = self.train_feature.to(device)

        in_dim = self.train_feature.shape[1]

        if preload is None:
            self.classifier = AdaptiveMetricModel(args)
            # setup classifier configuration
            self.classifier.setup(in_dim)
        else:
            self.classifier = AdaptiveMetricModel.load(preload)

        print("OML object created!")

        self.MetricClassifiers = []

        if preload is None:
            self.alpha = None
        else:
            self.alpha = self.classifier.alpha

    def generate(self):
        cls_a, cls_n = np.random.choice(self.labelSet, size=2, replace=False)
        if len(self.repository[cls_a]) == 1:
            if len(self.repository[cls_n]) > 1:
                tmp = cls_a
                cls_a = cls_n
                cls_n = tmp
            else:
                while True:
                    cls_n = np.random.choice(self.labelSet, size=1, replace=False)[0]
                    if len(self.repository[cls_n]) > 1:
                        break
                tmp = cls_a
                cls_a = cls_n
                cls_n = tmp

        x1_idx, x2_idx = np.random.choice(self.repository[cls_a], size=2, replace=False)
        x3_idx = np.random.choice(self.repository[cls_n], size=1, replace=False)
        return x1_idx, x2_idx, x3_idx

    def start(self, valid_feature=None, valid_label=None, evaluate_valid=False):
        if evaluate_valid:
            assert valid_feature is not None and valid_label is not None
        self.alpha = [1/(self.args.L + 1)] * (self.args.L + 1)

        if not os.path.exists(self.args.result_dir):
            os.makedirs(self.args.result_dir)
        writer2 = None
        if evaluate_valid:
            f2 = open(os.path.join(self.args.result_dir, self.args.result_file), 'w+')
            writer2 = csv.DictWriter(f2, fieldnames=['Iteration', 'Accuracy'])
            writer2.writeheader()

        with open(os.path.join(self.args.result_dir, self.args.alpha_file), 'w+') as f:
            print("Alpha Result File Path: ", os.path.join(self.args.result_dir, self.args.alpha_file))
            writer = csv.DictWriter(f, fieldnames=['Iteration', 'Alpha'])
            writer.writeheader()

            for i in progressbar.progressbar(range(self.args.iterations), redirect_stdout=True):
                idx1, idx2, idx3 = self.generate()
                x1, x2, x3 = self.train_feature[idx1], self.train_feature[idx2], self.train_feature[idx3]
                l1, l2, l3 = self.train_label[idx1], self.train_label[idx2], self.train_label[idx3]

                _, self.alpha, output_str = self.classifier.fit_triplet(x1, x2, x3, self.metricLogger)

                # verbose
                if self.args.verbose:
                    output_str = '#'*50+" iteration " + str(i) + " " + "#"*50 + '\n' + output_str
                    print(output_str)

                # evaluate
                if evaluate_valid:
                    if i % 1000 == 0:
                        self.build_knn_classifier(args=self.args)
                        p_label, p_prob = self.predict(valid_feature)
                        acc = np.sum(p_label == valid_label)/len(valid_label)
                        writer2.writerow({
                            'Iteration': i,
                            'Accuracy': acc
                        })
                writer.writerow({
                    'Iteration': i,
                    'Alpha': ','.join([str(each) for each in self.alpha])
                })
                # save model
                if i % self.args.save_segment == 0:
                    self.classifier.save('{}/{}.pth'.format(self.checkpoint_dir, i//self.args.save_segment))

            for i in range(len(self.alpha)):
                self.alpha[i] = 0
                self.classifier.alpha[i] = 0
            self.alpha[0] = 1  # index can be any one between 0 to self.args.L
            self.classifier.alpha[0] = 1

            self.classifier.save('{}/{}.pth'.format(self.checkpoint_dir, 'final'))

    def build_knn_classifier(self, train_feature=None, train_label=None, k=5, cuda=True, args=None):
        self.MetricClassifiers = []
        device = torch.device("cuda:"+self.args.gpu_id if self.args.cuda else "cpu")
        if train_feature is None:
            train_feature = self.train_feature
        else:
            train_feature = torch.from_numpy(train_feature)
            if self.args.cuda:
                train_feature = train_feature.to(device)

        if train_label is None:
            train_label = self.train_label

        for i in range(self.args.L+1):
            # self.MetricClassifiers.append(KNeighborsClassifier(n_neighbors=k, metric=Utility.metric()))
            self.MetricClassifiers.append(GpuKNN(n_neighbors=k, args=args, cuda=cuda))

        # evaluation after training
        for layer in range(self.args.L+1):
            # transform into embedding space
            transformed_feature = self.classifier.transform(train_feature, layer_idx=layer)
            self.MetricClassifiers[layer].fit(transformed_feature, train_label)
        return self

    def predict(self, x):
        x = torch.from_numpy(x)
        device = torch.device("cuda:"+self.args.gpu_id if self.args.cuda else "cpu")
        if self.args.cuda:
            x = x.to(device)
        predict_probs = []
        for layer in range(self.args.L+1):
            transformed_x = self.classifier.transform(x, layer_idx=layer)
            predict_prob = self.MetricClassifiers[layer].predict(transformed_x)
            predict_probs.append(predict_prob)

        final_p_label = []
        final_p_prob = []
        # generate predictions
        for i in range(len(x)):
            tmp = {}
            for layer in range(self.args.L+1):
                p = predict_probs[layer][i]

                for l in p:
                    if l not in tmp:
                        tmp[l] = self.alpha[layer] * p[l]
                    else:
                        tmp[l] += self.alpha[layer] * p[l]

            p_label, p_prob = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[0]
            final_p_label.append(p_label)
            final_p_prob.append(p_prob)

        final_p_label = np.array(final_p_label)
        final_p_prob = np.array(final_p_prob)
        return final_p_label, final_p_prob





