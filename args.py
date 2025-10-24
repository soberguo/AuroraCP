import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False)
    parser.add_argument('--ckpt', default='sota_ckpt/epoch_020.pt', type=str) 
    parser.add_argument('--ablation', default='2t', type=str)
    parser.add_argument('--save_dir', default='ablation_2t', type=str)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--roll_step', default=4, type=int)





    return parser.parse_args()