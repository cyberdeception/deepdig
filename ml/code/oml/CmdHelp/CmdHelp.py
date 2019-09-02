import argparse
import os


class CmdHelper:
    parser = argparse.ArgumentParser(description='Neural Metric Stream Learning')

    """ Online Metric Classifier """
    parser.add_argument('--base-dir', default='.')
    parser.add_argument('--checkpoint-dir', default='checkpoint',
                        help='folder to output model checkpoints')
    parser.add_argument('--log-dir', default='logs',
                        help='folder to store log files')
    parser.add_argument('--embedding-size', type=int, default=50, metavar='ES',
                        help='Dimensionality of the embedding (default: 50)')
    parser.add_argument('--margin',type=float, default=1.0, metavar='MARGIN',
                        help='the margin value for the triplet loss function (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--nn-structure', type=str, default='200', metavar='NNS',
                        help='Neural Network hidden layer size')
    parser.add_argument('--L', type=int, default=1, help='maximum number of hidden layers')
    parser.add_argument('--normalize', action='store_false', default=True, help='Normalize or not')
    parser.add_argument('--lr', type=float, default=0.3, help='learning rate (default: 0.3)')
    parser.add_argument('--lr-decay', type=float, default=1e-4, help='learning rate decay ratio (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=0., help='weight decay (default: 0.)')
    parser.add_argument('--optimizer', default='adam', type=str, help='The optimizer to use (default: ADAM)')
    parser.add_argument('--smooth', type=float, default=0.1, help='smooth factor')
    parser.add_argument('--beta-prime', type=float, default=0.99, help='decay factor for alpha update')
    # parser.add_argument('--lamb2', type=float, default=1.0, help='importance of angle distance')
    # parser.add_argument('--lamb1', type=float, default=1.0, help='Importance of Euclidean distance')
    parser.add_argument('--tau', type=float, default=0.5, help='effective radius')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--verbose', action='store_false', default=False)

    """ OML Settings """
    parser.add_argument('--train-ratio', type=float, default=0.6, help='ratio of training data')
    parser.add_argument('--valid-ratio', type=float, default=0.2, help='ratio of validation data')
    parser.add_argument('--iterations', type=int, default=5*10**4, help='number of iterations')
    parser.add_argument('--result-dir', type=str, default='result')
    parser.add_argument('--result-file', type=str, default='result.csv')
    parser.add_argument('--alpha-file', type=str, default='alpha.csv')
    parser.add_argument('--save-segment', type=int, default=1000)
    parser.add_argument('--trainpath', type=str, default=".")
    parser.add_argument('--testpath', type=str, default=".")
    parser.add_argument('--classes', type=str, default="")
    @staticmethod
    def get_parser():
        return CmdHelper.parser
