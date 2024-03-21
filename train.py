import json
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


def train_one_epoch(epoch_num, model, train_loader, preprocess_fn, optimizer, loss_fn, batch_size):
    model.train()
    batch_running_loss = 0
    accurate_predictions = 0
    for index, batch in enumerate(train_loader):
        batch = to_device(batch)
        x, y = batch
        x = preprocess_fn(x)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        batch_running_loss += loss.item()
        accurate_predictions += (torch.argmax(prediction, dim=1) == torch.argmax(y, dim=1)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step = (epoch_num - 1) * len(train_loader) + index + 1
        if step % 10 == 0:
            report(batch_running_loss / 10, accurate_predictions / (10 * batch_size), step)
            batch_running_loss = 0
            accurate_predictions = 0


def evaluate(epoch_num, model, test_loader, preprocess_fn, loss_fn, batch_size):
    model.eval()
    test_loss = 0
    accurate_predictions = 0
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            batch = to_device(batch)
            x, y = batch
            x = preprocess_fn(x)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            test_loss += loss.item()
            accurate_predictions += (torch.argmax(prediction, dim=1) == torch.argmax(y, dim=1)).sum().item()
            batch_num = index + 1
            if batch_num % 10 == 0:
                print(
                    f'Epoch: {epoch_num}, '
                    f'Batches: up to {batch_num}, '
                    f'Test Loss: {test_loss / batch_num}, '
                    f'Test Accuracy: {round(accurate_predictions / (batch_num * batch_size) * 100, 2)} %')
    print(
        f'Epoch: {epoch_num}, '
        f'Overall Test Loss: {test_loss / len(test_loader)}, '
        f'Overall Test Accuracy: {round(accurate_predictions / (batch_size * len(test_loader)) * 100, 2)} %')


def report(loss, accuracy, step):
    summary_writer.add_scalar(f'Train/Loss', loss, step)
    summary_writer.add_scalar(f'Train/Accuracy', accuracy, step)
    print(f'Training Step: {step}, Loss: {loss}, Accuracy: {round(accuracy * 100, 2)} %')


def checkpoint(epoch_num, model, args_dict):
    path = os.path.join(args_dict.get('checkpoints_dir'), f'epoch-{epoch_num}')
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, f'model.pt'))
    with open(os.path.join(path, 'args.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)


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

    args_dict = vars(args)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch, model, train_loader, data_preprocessor, optimizer, loss_fn, args.batch_size)
        checkpoint(epoch, model, args_dict)
        evaluate(epoch, model, test_loader, data_preprocessor, loss_fn, args.batch_size)

    summary_writer.close()


if __name__ == '__main__':
    main()
