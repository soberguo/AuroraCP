import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False)
    parser.add_argument('--ckpt', default='aurora_checkpoints/epoch_002.pt', type=str)



    return parser.parse_args()