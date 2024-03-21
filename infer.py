import argparse
import json
import os

import torch

from src.dataset.dataset import Dataset
from src.dataset.preprocess import init_data_preprocessor
from src.util.device import to_device


def parse_args():
    parser = argparse.ArgumentParser(description='Keyword Transformer in PyTorch (Inference)')

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file.')

    return parser.parse_args()


def main():
    args = parse_args()

    model = torch.load(os.path.join(args.model, 'model.pt'))
    to_device(model)
    model.eval()

    with open(os.path.join(args.model, 'args.json')) as file:
        training_args = argparse.Namespace(**json.load(file))

    labels = ['unknown', *training_args.wanted_words.split(',')]

    data_preprocessor = init_data_preprocessor(training_args)
    waveform = Dataset.load_audio(args.audio)
    x = to_device(waveform)
    x = x[None, :]  # simulating a batch
    x = data_preprocessor(x)

    with torch.no_grad():
        prediction = model(x)
        print(labels[prediction.argmax().item()])


if __name__ == '__main__':
    main()
