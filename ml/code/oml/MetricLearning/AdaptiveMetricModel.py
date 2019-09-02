from MetricLearning.model import CombNet
from torch.autograd import Variable
import torch
from copy import deepcopy
import numpy as np


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2, eps=1e-6):
        assert x1.size() == x2.size()
        diff = torch.abs(x1-x2)
        out = torch.pow(diff+eps, self.norm).sum(dim=-1)
        euc = torch.pow(out, 1./self.norm)
        return euc


class PotentialLoss(torch.nn.Module):
    def __init__(self, args):
        super(PotentialLoss, self).__init__()
        self.args = args

    def forward(self, threshold, dist, att):
        if att:
            tmp1 = 1./(2*1+0-threshold)
            tmp2 = -threshold * tmp1
            tmp = tmp1 * dist + tmp2
            return torch.clamp(tmp, min=0.)
        else:
            tmp1 = -1./threshold
            tmp = tmp1 * dist + 1
            return torch.clamp(tmp, min=0.)


class AdaptiveMetricModel(object):
    def __init__(self, args, ):
        if args:
            self.args = args

        self.l2_dist = PairwiseDistance(2)
        self.potLoss = PotentialLoss(args)

        self.model = None
        self.L = self.args.L
        self.alpha = np.array([1./(self.L+1)]*(self.L+1))
        self.optimizer = None
        self.in_dim = None

        # compute some constants
        att_1 = self.args.tau/(np.exp(2*1)-1)
        att_2 = -att_1
        push_1 = ((2*1)-(2-self.args.tau)*1)/(1-np.exp(-(2*1)))
        push_2 = push_1 + (2-self.args.tau)*1
        self.att_1 = float(att_1)
        self.att_2 = float(att_2)
        self.push_1 = float(push_1)
        self.push_2 = float(push_2)

        # accumulative loss
        self.acc_loss = np.zeros(self.args.L+1)
        self.count = 0

    def setup(self, in_dim):
        print("input dimension: ", in_dim)
        for key in self.args.__dict__:
            print("{} : {}".format(key, self.args.__dict__[key]))

        self.model = CombNet(self.args, in_dim)
        self.in_dim = in_dim
        if self.args.cuda:
            self.model.cuda()
        self.optimizer = self.create_optimizer(self.args.lr)

    def create_optimizer(self, lr):
        optimizer = None
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, dampening=0.9,
                                        weight_decay=self.args.wd, nesterov=False)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.wd, amsgrad=True)
        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr, lr_decay=self.args.lr_decay,
                                            weight_decay=self.args.wd)
        elif self.args.optimizer == 'nesterov':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, dampening=0.,
                                        weight_decay=self.args.wd, nesterov=True)
        return optimizer

    def fit_triplet(self, x1, x2, x3, metric_logger, output_str=""):
        device = torch.device("cuda:" + self.args.gpu_id if self.args.cuda else "cpu")
        if self.args.normalize:
            x1, x2, x3 = x1/255, x2/255, x3/255
        if self.args.cuda:
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

        x1, x2, x3 = Variable(x1, requires_grad=True), Variable(x2, requires_grad=True), Variable(x3,
                                                                                                  requires_grad=True)

        # forward to embedding space
        out_x1, out_x2, out_x3 = self.model.forward(x1), self.model.forward(x2), \
                                 self.model.forward(x3)  # (n_layer, 1, embedding size)

        # compute distance between x1 and x2
        dist = self.l2_dist.forward(out_x1, out_x2)  # (n_layer, ) distance for each layer
        dist2 = self.l2_dist.forward(out_x1, out_x3)

        # compute threshold
        # similar pairs
        threshold1 = self.att_1 * torch.exp(dist) + self.att_2
        # dissimialr pairs
        threshold2 = self.push_1 * (-torch.exp(-dist2)) + self.push_2
        # compute loss value
        loss1 = self.potLoss.forward(threshold1, dist, 1)
        loss2 = self.potLoss.forward(threshold2, dist2, 0)

        # find loss
        loss = loss1/(loss1 + loss2) * loss1 + loss2/(loss1 + loss2) * loss2

        # check for nan
        idx = torch.isnan(loss)
        loss[idx] = (loss1[idx] + loss2[idx])/2

        loss_numbers = []

        for layer in range(self.args.L + 1):
            loss_value = loss[layer].data.cpu().numpy()[0]
            dist1_number = dist[layer].data.cpu().numpy()[0]
            dist2_number = dist2[layer].data.cpu().numpy()[0]
            if self.args.verbose:
                output_str += "Loss of Triplet: Layer {} -- Loss: {:.5f} -- Distance: {:.5f} - {:.5f}\n".format(
                    layer, loss_value, dist1_number, dist2_number)
            loss_numbers.append(loss_value)

        if self.args.verbose:
            output_str += "alpha: {}\n".format(self.alpha, )

        loss_numbers = np.array(loss_numbers)

        # compute accumulative loss
        self.acc_loss += loss_numbers
        self.count += 1

        if self.args.verbose:
            for layer in range(self.args.L + 1):
                output_str += 'Accumulative Loss: Layer {} -- Loss: {:.5f}\n'.format(layer,
                                                                                     self.acc_loss[layer]/self.count)

        print(self.model.nn[1])
        # compute gradient
        weight_grad = {}
        for i in range(self.args.L + 1):
            self.optimizer.zero_grad()
            loss[i].backward(retain_graph=True)

            # store gradient
            if i == 0:
                weight_grad[i] = {
                    'theta1': deepcopy(self.model.nn[i][0].weight.grad),
                }
            else:
                weight_grad[i] = {
                    'theta1': deepcopy(self.model.nn[i][2].weight.grad),
                    'theta1idx': i * 3,
                }
                tmp = []
                for j in range(1, self.args.L+1):
                    tmp.append(deepcopy(self.model.nn[j][0].weight.grad))
                weight_grad[i]['ws'] = tmp

        # merge gradients
        self.optimizer.zero_grad()
        for layer in range(self.args.L+1):
            if layer == 0:
                self.optimizer.param_groups[0]['params'][0].grad = weight_grad[layer]['theta1'] * self.alpha[
                    layer]
            else:
                self.optimizer.param_groups[0]['params'][layer * 3].grad = weight_grad[layer][
                                                                               'theta1'] * \
                                                                           self.alpha[layer]
                for j in range(layer, self.args.L+1):
                    self.optimizer.param_groups[0]['params'][layer * 3 - 2].grad += (
                            weight_grad[j]['ws'][layer - 1] *
                            self.alpha[j])

        # update
        self.optimizer.step()

        # find minimum loss
        min_loss = np.min(loss_numbers)
        if self.args.beta_prime ** min_loss * np.log(min_loss) > self.args.beta_prime - 1:
            self.alpha *= self.args.beta_prime ** loss_numbers
        else:
            self.alpha *= (1 - (1 - self.args.beta_prime) * loss_numbers)
        self.alpha = np.maximum(self.alpha, self.args.smooth / (self.args.L+1))
        self.alpha = self.alpha / np.sum(self.alpha)

        # log loss value
        for layer in range(self.args.L+1):
            metric_logger.log_value('loss' + str(layer) + ': ', loss_numbers[layer])

        return None, self.alpha, output_str

    def transform(self, features, layer_idx, return_numpy=True):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        device = torch.device("cuda:"+self.args.gpu_id if self.args.cuda else "cpu")
        features = features.to(device)
        transformed_features = self.model.forward(features, layer_idx=layer_idx)
        transformed_features = transformed_features[0,:,:]
        if return_numpy:
            transformed_features = transformed_features.detach().cpu().numpy()
        return transformed_features

    def save(self, path):
        torch.save(
            {
                'args': self.args,
                'state_dict': self.model.state_dict(),
                'alpha': self.alpha,
                "in_dim": self.in_dim
            },
            path
        )

    @staticmethod
    def load(path):
        parameters = torch.load(path)

        net = CombNet(parameters['args'], parameters['in_dim'])
        net.load_state_dict(parameters['state_dict'])

        model = AdaptiveMetricModel(parameters['args'])
        model.model = net
        model.in_dim = parameters['in_dim']
        model.alpha = parameters['alpha']

        return model
