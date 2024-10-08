import json
import argparse
from omegaconf import OmegaConf


def get_parameters(para_path = "configs/test.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=para_path)
    parser.add_argument('-t', '--task_id', type=str, default='debug') # not used
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    

    args = parser.parse_args()

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    if args.debug:
        args.train.batch_size = 4

    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args)
