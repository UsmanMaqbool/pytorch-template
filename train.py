import argparse
from parse_config import ConfigParser

import collections

import torch
import numpy as np


def main(config):
    logger = config.get_logger('train')



if __name__ == '__main__':
    args = argparse.ArgumentParser(description = 'Pytorch NetVLAD')
    # Add some actions
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    args_custom = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    
    # add / update config related to dataset, saved path, dimension, cache
    config = ConfigParser.from_args(args, args_custom)
    main(config)