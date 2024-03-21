import torch
import torchaudio
import torchaudio.transforms as T

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
    time_masking = T.TimeMasking(time_mask_param=args.time_mask_max_size)
    frequency_masking = T.FrequencyMasking(freq_mask_param=args.frequency_mask_max_size)

    def preprocess(x):
        x = mfcc(x)
        for _ in range(args.time_masks_number):
            x = time_masking(x)
        for _ in range(args.frequency_masks_number):
            x = frequency_masking(x)
        return [transform(x)[:, 0, :, :] for transform in transforms]

    return preprocess
