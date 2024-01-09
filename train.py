import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset.dataset import Dataset
from src.dataset.preprocess import init_data_preprocessor
from src.model.keyword_transformer import KWSTransformer
from src.util.args_parser import parse_args
from src.util.device import to_device

summary_writer = SummaryWriter()


def train_one_epoch(epoch_num, model, train_loader, preprocess_fn, optimizer, loss_fn):
    model.train()
    epoch_loss = 0
    batch_running_loss = 0
    for index, batch in enumerate(train_loader):
        batch = to_device(batch)
        x, y = batch
        x = preprocess_fn(x)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        epoch_loss += loss.item()
        batch_running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index % 10 == 9:
            print(f'Batch: {index + 1}, loss: {batch_running_loss / 10}')
            batch_running_loss = 0
    print_loss('Training loss', epoch_loss / len(train_loader), epoch_num)


def evaluate(epoch_num, model, test_loader, preprocess_fn, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch)
            x, y = batch
            x = preprocess_fn(x)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            test_loss += loss.item()
    print_loss('Test loss', test_loss / len(test_loader), epoch_num)


def print_loss(tag, loss, epoch):
    summary_writer.add_scalar(tag, loss, epoch)
    print(f'Epoch: {epoch}, {tag}: {loss}')


def checkpoint(epoch_num, model, checkpoint_dir):
    torch.save(model, os.path.join(checkpoint_dir, f'epoch-{epoch_num}.pt'))


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    model = KWSTransformer(args)
    to_device(model)

    train_loader = DataLoader(Dataset(args, 'training'), batch_size=args.batch_size)
    test_loader = DataLoader(Dataset(args, 'testing'), batch_size=args.batch_size)
    data_preprocessor = init_data_preprocessor(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch, model, train_loader, data_preprocessor, optimizer, loss_fn)
        evaluate(epoch, model, test_loader, data_preprocessor, loss_fn)
        checkpoint(epoch, model, args.checkpoints_dir)


if __name__ == '__main__':
    main()
