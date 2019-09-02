import torch
from collections import Counter
import numpy as np


class GpuKNN(object):
    def __init__(self, n_neighbors, args, cuda=True):
        self.k = n_neighbors
        self.trainX = None
        self.trainY = None
        self.cuda = cuda
        self.args = args

    def fit(self, x, y=None):
        x = torch.from_numpy(x)
        device = torch.device("cuda:"+self.args.gpu_id if self.cuda else "cpu")
        if self.cuda:
            x = x.to(device)
        self.trainX = x
        self.trainY = y

    def neighbors(self, x, n, eps=1e-6):
        x = torch.from_numpy(x)
        device = torch.device("cuda:"+self.args.gpu_id if self.cuda else "cpu")
        if self.cuda:
            x = x.to(device)
        point = x
        # euclidean distance
        tmp1 = point - self.trainX
        diff = torch.abs(tmp1)
        out = torch.pow(diff+eps, 2).sum(dim=-1)
        euc = torch.pow(out, 1./2)
        # complete distance
        dist = euc

        # find top k
        values, ind = torch.topk(dist, n+1, largest=False)
        # judge if point itself is included
        ind = ind[1:]
        return ind

    def predict(self, x, eps=1e-6):
        print('\n')
        n_test = len(x)

        x = torch.from_numpy(x)
        if self.cuda:
            x = x.cuda()

        prediction_probs = []
        for i in range(n_test):
            print("test instance %s out of %s" % (i, n_test), end='\r')
            point = x[i]
            # euclidean distance
            tmp1 = point - self.trainX
            diff = torch.abs(tmp1)
            out = torch.pow(diff+eps, 2).sum(dim=-1)
            euc = torch.pow(out, 1./2)
            # complete distance
            dist = euc

            # find top k
            values, ind = torch.topk(dist, self.k, largest=False)
            # normalize distances as probabilities
            # max distance and min distance
            d_max, d_min = torch.max(values), torch.min(values)
            probs = torch.exp(-(values - d_min)/(d_max-d_min))

            # find prediction probability for each label
            ind = ind.cpu().numpy()
            labels = self.trainY[ind]
            label_prob = {}
            for j, each in enumerate(labels):
                if each not in label_prob:
                    label_prob[each] = 0
                label_prob[each] += probs[j]
            prediction_probs.append(label_prob)
        return prediction_probs



