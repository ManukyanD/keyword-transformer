import os
import random
from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from src.util.constants import SAMPLING_RATE


class Dataset(SPEECHCOMMANDS):
    def __init__(self, args, subset):
        super().__init__(root=args.data_dir, download=True, subset=subset)
        self.wav_root = os.path.join(args.data_dir, 'SpeechCommands', 'speech_commands_v0')
        self.UNKNOWN_KEY = '_unknown_'

        self.words_paths = self.initialize_words_paths()
        self.words = self.words_paths.keys()

        self.wanted_words = [self.UNKNOWN_KEY, *args.wanted_words.split(',')]
        self.unwanted_words = [word for word in self.words if word not in self.wanted_words]

        label_count = len(self.wanted_words)
        self.wanted_words_labels = dict(
            zip(self.wanted_words,
                [torch.nn.functional.one_hot(torch.tensor(index), num_classes=label_count).float() for index in
                 range(label_count)]))

    def initialize_words_paths(self):
        words_paths = {}
        for path in self._walker:
            word = Path(path).parts[-2]
            words_paths.setdefault(word, []).append(path)
        return words_paths

    def __getitem__(self, _):
        word = random.choice(self.wanted_words)
        label = self.wanted_words_labels[word]
        if word == self.UNKNOWN_KEY:
            word = random.choice(self.unwanted_words)
        waveform, sample_rate = torchaudio.load(random.choice(self.words_paths[word]))
        waveform = torch.nn.functional.pad(waveform, pad=(0, SAMPLING_RATE - waveform.shape[1]), value=0)
        return waveform, label
