import torch.nn as nn
import torch
from collections import OrderedDict


class CombNet(nn.Module):
    def __init__(self, args, in_dim):
        super(CombNet, self).__init__()

        self.args = args
        # find maxiumum number of hidden layers
        L = args.L

        # find neural network structure of these hidden layers
        hidden_size = [int(each.strip()) for each in args.nn_structure.split(',')]

        # find embedding size
        embedding_size = args.embedding_size

        """ Build Neural Network Structure """
        layers = OrderedDict()
        last_hidden_size = in_dim

        for i in range(L+1):
            layer = OrderedDict()

            # add layers
            if i == 0:
                layer['0'] = nn.Linear(in_dim, embedding_size, bias=False)  # input-embedding
                # layer['1'] = nn.Linear(embedding_size, n_class, bias=False)  # embedding - classification
                #print(layer['0'].weight.data.size(), torch.eye(in_dim).size())
                #layer['0'].weight.data.copy_(torch.eye(in_dim))# just added april 2
            else:
                layer['0'] = nn.Linear(last_hidden_size, hidden_size[i-1], bias=True)  # hidden-hidden
                layer['1'] = nn.ReLU()  # activation
                layer['2'] = nn.Linear(hidden_size[i-1], embedding_size, bias=False)  # hidden-embedding
                # layer['3'] = nn.Linear(embedding_size, n_class, bias=False)  # embedding-classification
                last_hidden_size = hidden_size[i-1]  # change hidden size
            layers[str(i)] = nn.Sequential(layer)
        self.nn = nn.Sequential(layers)
        print("\n Network Structure: \n", self.nn)

    def forward(self, x, layer_idx=None):
        if len(x.size()) == 1:
            x = x.view(1, -1)
        x = x.float()
        if layer_idx is not None:
            if layer_idx == 0:
                x = self.nn[0][0](x)
            else:
                for i in range(1, layer_idx+1):
                    x = self.nn[i][0](x)  # hidden-hidden
                    x = self.nn[i][1](x)  # activation
                x = self.nn[layer_idx][2](x)  # hidden-embedding
            x = self.l2_norm(x)
            return x.unsqueeze(0)
        else:
            outs = []
            # add 0 layer
            out = self.nn[0][0](x)  # input-embedding
            # out = out.unsqueeze(0)
            outs.append(self.l2_norm(out).unsqueeze(0))

            # add following layers
            for i in range(1, self.args.L+1):
                x = self.nn[i][0](x)  # hidden-hidden
                x = self.nn[i][1](x)  # activation
                out = self.nn[i][2](x)  # hidden-embedding
                # out = out.unsqueeze(0)
                out = self.l2_norm(out)
                outs.append(out.unsqueeze(0))  # add to embedding output
            outs = torch.cat(outs, dim=0)
            return outs

    def l2_norm(self, input):
        input_size = input.size()
        temp = torch.pow(input, 2)
        if len(input.size()) > 1:
            normp = torch.sum(temp, -1).add_(1e-10)
        else:
            normp = torch.sum(temp).add_(1e-10)
        norm = torch.sqrt(normp)
        if len(input.size()) > 1:
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        else:
            _output = torch.div(input, norm.expand_as(input))

        output = _output.view(input_size)

        return output
