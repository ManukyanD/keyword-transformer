import torch
import torchaudio

from src.util.device import to_device


def init_patch_extractor(args):
    def extract(x):
        batch_size, *_ = x.shape
        return (x
                .unfold(2, args.patch_size_f, args.patch_size_f)
                .unfold(3, args.patch_size_t, args.patch_size_t)
                .permute(0, 2, 3, 1, 4, 5)
                .contiguous()
                .view(batch_size, args.num_patches, -1))

    return extract


def init_data_preprocessor(args):
    transforms = []
    if args.attn_type == 'time' or args.attn_type == 'both':
        transforms.append(lambda x: torch.permute(x, (0, 1, 3, 2)))
    if args.attn_type == 'freq' or args.attn_type == 'both':
        transforms.append(lambda x: x)
    elif args.attn_type == 'patch':
        transforms.append(init_patch_extractor(args))

    mfcc = to_device(torchaudio.transforms.MFCC(
        melkwargs={"n_fft": args.n_fft, "hop_length": args.fft_hop_length, "normalized": True, "center": False}))

    def preprocess(x):
        x = mfcc(x)
        return [transform(x).squeeze() for transform in transforms]

    return preprocess
