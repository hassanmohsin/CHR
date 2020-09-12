import argparse
import json

import torch
from torch.nn.modules.loss import _WeightedLoss

from CHR.engine import MultiLabelMAPEngine
from CHR.models import resnet101_CHR
from CHR.ray import XrayClassification


class LoadConfig:
    def __init__(self, json_path):
        self.__dict__ = json.load(open(json_path))

    def __iter__(self):
        return self

    def __next__(self):
        for k in self.__dict__.keys():
            yield k


def binary_cross_entropy(inputs, target, eps=1e-10):
    # if not (target.size() == input.size()):
    #     warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
    #                   "Please ensure they have the same size.".format(target.size(), input.size()))
    # if input.nelement() != target.nelement():
    #     raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
    #                      "!= input nelement ({})".format(target.nelement(), input.nelement()))
    #
    # if weight is not None:
    #     new_size = _infer_size(target.size(), weight.size())
    #     weight = weight.expand(new_size)
    #     if torch.is_tensor(weight):
    #         weight = Variable(weight)

    inputs = torch.sigmoid(inputs)
    return -(target * torch.log(inputs + eps) + (1 - target) * torch.log(1 - inputs + eps))


class MultiLabelSoftMarginLoss(_WeightedLoss):

    def forward(self, input, target):
        return binary_cross_entropy(input, target)


def main_ray(args):
    # define dataset
    train_dataset = XrayClassification(args.data, 'train', args.subset)
    val_dataset = XrayClassification(args.data, 'test', args.subset)

    # load model
    model = resnet101_CHR(args.num_classes, pretrained=args.pretrained)

    # define loss function (criterion)
    criterion = MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    engine = MultiLabelMAPEngine(args)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHR Training')
    parser.add_argument('-c', '--config', metavar='DIR', required=True,
                        help='config file')
    args = parser.parse_args()
    config = LoadConfig(args.config)
    # config = LoadConfig("../configs/train_config.json")
    main_ray(config)
