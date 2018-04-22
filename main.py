from model import *
from char_list import *
from loader import *

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


CUDA_AVAILABLE = torch.cuda.is_available()

def decode(seq):
    return ''.join([DECODE_MAP[c] for c in seq])

def to_cuda(*tensors):
    if CUDA_AVAILABLE:
        tensors = tuple(tensor.cuda() for tensor in tensors)
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tensors

def load_data(name):
    return (np.load('{}.npy'.format(name)),
            np.load('{}_encode.npy'.format(name)))

def unigram_logits(stat, smoothing=.01):
    char_size = len(stat)
    char_sum = stat[:, 1].sum()
    p = stat[:, 1] / char_sum
    p_smoothed = p * (1 - smoothing) + smoothing / char_size
    return np.log(p_smoothed)

def routine(args, model, loader, optimizer, criterion, epoch, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses = AverageMeter()
    start = time.time()
    for i, (seq, seq_len, label_in, label_out, label_mask, _) in enumerate(loader):
        seq, label_in, label_out, label_mask = to_cuda(seq, label_in, label_out, label_mask)
        logits = model(seq, seq_len, label_in).transpose(0, 1).contiguous()  # (T, N, char_size) -> (N, T, char_size)
        loss_raw = criterion(logits, label_out)   # (N, T)
        loss = (loss_raw * label_mask).clamp(min=1e-9).sum(dim=1).mean()   # use clamp to make dot product num_stable

        # update metrics
        losses.update(loss.data.cpu()[0], len(seq_len))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % args.print_freq == 0:
            running_time = time.time() - start
            start = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'time: {time:.4f}'.format(epoch, i, len(loader), loss=losses, time=running_time))
    return losses.avg

def train(args):
    print('Train mode')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    stat_encode = np.load('stat_encode.npy')
    projection_bias = torch.FloatTensor(unigram_logits(stat_encode))
    train_data = Dataset(*load_data('train'))
    valid_data = Dataset(*load_data('dev'))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    if args.model:
        print('loading model: {}'.format(args.model))
        model = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        model = LASModel(feed_forward_ratio=args.feed_forward_ratio, projection_bias=projection_bias)
        model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = CrossEntropyLoss3D(reduce=False)
    if CUDA_AVAILABLE:
        model.cuda()
        criterion.cuda()

    for epoch in range(args.epoch):
        print('-' * 10 + 'epoch {}: train'.format(epoch) + '-' * 10)
        loss = routine(args, model, train_loader, optimizer, criterion, epoch, True)
        print('train epoch {}:\tloss={:.4f}'.format(epoch, loss))
        print('-' * 10 + 'epoch {}: evaluate'.format(epoch) + '-' * 10)
        loss = routine(args, model, valid_loader, optimizer, criterion, epoch, False)
        print('valid epoch {}:\tloss={:.4f}'.format(epoch, loss))
        if epoch % args.save_freq == 0:
            filename = '{}_{:.4f}.pt'.format(epoch, loss)
            path = os.path.join(args.save_dir, filename)
            print('Saving new model to: {}'.format(path))
            torch.save(model, path)

def generate_sequence(model, seq, seq_len, num_seq=10):
    result = []
    label_in = Variable(seq.data.new([[SOS_IDX]]).long())
    for _ in range(num_seq):
        out = model(seq, seq_len, label_in, predict=True)
        result.append(out.data.cpu().numpy())
    return np.array(result)

def test(args):
    print('Test mode')
    xtest = np.load('test.npy')
    test_loader = TestLoader(xtest)
    print('loading model: {}'.format(args.model))
    model = torch.load(args.model, map_location=lambda storage, loc: storage).eval()
    criterion = CrossEntropyLoss3D(reduce=False)
    if CUDA_AVAILABLE:
        model.cuda()
        criterion.cuda()

    result = []
    total_loss = 0.
    for i, (seq, seq_len) in enumerate(test_loader):
        seq = to_cuda(seq)
        start = time.time()
        candidate = generate_sequence(model, seq, seq_len, args.num_seq)
        # (L, 1, T) ->(L, N, T) -> (N, L, T)
        seq = seq.data.expand(-1, args.num_seq, -1).transpose(0, 1).cpu().numpy()
        eval_loader = DataLoader(seq, candidate, batch_size=args.batch_size, random=False)
        min_loss = 1e5
        best_seq = None
        for seq, seq_len, label_in, label_out, label_mask, label_len in eval_loader:
            seq, label_in, label_out, label_mask = to_cuda(seq, label_in, label_out, label_mask)
            logits = model(seq, seq_len, label_in).transpose(0, 1).contiguous()  # (T, N, char_size) -> (N, T, char_size)
            loss_raw = criterion(logits, label_out)  # (N, T)
            losses = (loss_raw * label_mask).clamp(min=1e-9).sum(dim=1).data.cpu().numpy()
            loss, idx = losses.min(), losses.argmin()
            if loss < min_loss:
                min_loss = loss
                best_seq = label_in.data.cpu()[idx, 1:label_len[idx]]
        total_loss += min_loss
        decoded = decode(best_seq)
        running_time = time.time() - start
        if args.verbose:
            print('seq: {}\tloss: {:.4f}\ttime: {:.4f}\n'
                  'str: {}\n'.format(i, min_loss, running_time, decoded))
        result.append(decoded)

    print('Average Loss: {:.4f}'.format(total_loss / len(result)))
    y = np.array([np.arange(len(result)), result]).T
    df = pd.DataFrame(y, columns=['Id', 'Predicted'])
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    filename = args.model.split('/')[1][:-3] + '.csv'
    path = os.path.join(args.test_dir, filename)
    print('Save result to: {}'.format(path))
    df.to_csv(path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=32)
    parser.add_argument('--model', '-m', dest='model', type=str, default=None)
    parser.add_argument('--print-freq', dest='print_freq', type=int, default=25)
    parser.add_argument('--epoch', dest='epoch', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='models')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', default=True)
    parser.add_argument('--min-loss',dest='min_loss', type=float, default=50)
    parser.add_argument('--save-freq', dest='save_freq', type=int, default=1)
    parser.add_argument('--feed-forward', dest='feed_forward_ratio', type=float, default=.1)

    # for testing
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--num-seq', dest='num_seq', type=int, default=32)
    parser.add_argument('--test-dir', dest='test_dir', type=str, default='result')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)
    args = parser.parse_args()

    CUDA_AVAILABLE = torch.cuda.is_available() and args.cuda
    print('Using CUDA: {}'.format(CUDA_AVAILABLE))

    print(args)

    if (args.test):
        test(args)
    else:
        train(args)
