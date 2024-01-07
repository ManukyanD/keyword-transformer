import torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(device, non_blocking=True)
