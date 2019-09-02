import os
import torch


class Utility(object):
    args = None

    @staticmethod
    def setup(args):
        Utility.args = args

    @staticmethod
    def check_args(parser):
        args = parser.parse_args()
        # set the device to use by setting CUDA_VISIBLE_DEVICES env variable in order to prevent any memory allocation
        # on unused GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        return args
