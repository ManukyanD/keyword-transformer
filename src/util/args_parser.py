import argparse
import math
import os.path

from src.util.constants import SAMPLING_RATE, AUDIO_CLIP_DURATION_MS


def update_args(args):
    args.label_count = len(args.wanted_words.split(',')) + 1
    args.fft_window_size = round(SAMPLING_RATE * args.window_size_ms / AUDIO_CLIP_DURATION_MS)
    args.n_fft = args.fft_window_size
    args.fft_hop_length = round(SAMPLING_RATE * args.window_stride_ms / AUDIO_CLIP_DURATION_MS)
    args.num_time_steps = math.ceil((SAMPLING_RATE - args.fft_window_size) / args.fft_hop_length) + 1
    args.num_freq_bins = args.n_mfcc

    assert args.num_freq_bins % args.patch_size_f == 0 and args.num_time_steps % args.patch_size_t == 0, \
        (f"Spectrogram dimensions ({args.num_freq_bins} and {args.num_time_steps}) "
         f"must be divisible by patch sizes ({args.patch_size_f} and {args.patch_size_t}, respectively).")

    args.num_patches_t = args.num_time_steps / args.patch_size_t
    args.num_patches_f = args.num_freq_bins / args.patch_size_f
    args.num_patches = args.num_patches_t * args.num_patches_f
    args.patch_size = args.patch_size_t * args.patch_size_f
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='Keyword Transformer in PyTorch')

    parser.add_argument('--data-dir', type=str, default=os.path.join('.', 'data'),
                        help='The directory to download data to (default: "./data").')
    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='The directory to save checkpoints to (default: "./checkpoints").')

    # learning args
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10).')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Base learning rate (default: 0.001).')
    parser.add_argument('--l2-weight-decay', type=float, default=0.1,
                        help='L2 weight decay for layers weights regularization (default: 0.1).')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512).')
    parser.add_argument('--wanted-words', type=str, default='yes,no,up,down,left,right,on,off,stop,go',
                        help='Words to use (others will be added to an unknown label, default: yes,no,up,down,left,right,on,off,stop,go).')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing epsilon (default: 0.1).')
    parser.add_argument('--window-size-ms', type=float, default=30.0,
                        help='How long each spectrogram timeslice is (default: 30).')
    parser.add_argument('--window-stride-ms', type=float, default=10.0,
                        help='How far to move in time between spectrogram time-slices (default: 10).')
    parser.add_argument('--n-mfcc', type=int, default=40,
                        help='Number of mfc coefficients to retain (default: 40).')

    # model args
    parser.add_argument('--num-layers', type=int, default=12,
                        help='The number of transformer layers (default: 12).')
    parser.add_argument('--embed-dim', type=int, default=192,
                        help='Embedding dimension (default : 192).')
    parser.add_argument('--ff-dim', type=int, default=768,
                        help='Transformer ff dimension (default: 768).')
    parser.add_argument('--num-heads', type=int, default=3,
                        help='The number of attention heads (default : 3).')
    parser.add_argument('--dropout', type=int, default=0.,
                        help='Dropout value (default : 0.1).')
    parser.add_argument('--attn-type', type=str, default='both', choices=['time', 'freq', 'both', 'patch'],
                        help='Domain for attention: time, freq, both or patch (default: "time").')
    parser.add_argument('--patch-size-t', type=int, default=1,
                        help='Time steps in patch (default: 1).')
    parser.add_argument('--patch-size-f', type=int, default=40,
                        help='Frequency steps in patch (default: 40).')
    parser.add_argument('--prenorm',action='store_true',
                        help='Use prenorm instead of postnorm.')
    parser.add_argument('--approximate-gelu', action='store_true',
                        help='Use approximate GELU activation.')

    return update_args(parser.parse_args())
