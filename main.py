from model import *
from char_list import *

import argparse
import os
import time
import numpy as np
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
print('Using CUDA: {}'.format(CUDA_AVAILABLE))


def to_cuda(*tensors):
    if CUDA_AVAILABLE:
        return tuple(tensor.cuda() for tensor in tensors)
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
    for i, (seq, seq_len, label_in, label_out, label_mask) in enumerate(loader):
        seq, label_in, label_out, label_mask = to_cuda(seq, label_in, label_out, label_mask)
        seq_len = seq_len.numpy()   # for pack_padded_sequence
        logits = model(seq, seq_len, label_in).transpose(0, 1).contiguous() # (T, N, char_size) -> (N, T, char_size)
        loss_raw = criterion(logits, label_out)   # (N, T)
        loss = (loss_raw * label_mask).clamp(min=1e-9).sum(dim=1).mean()   # use clamp to make the dot product num_stable

        # update metrics
        losses.update(loss.data.cpu()[0], args.batch_size)

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

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    xtrain, ytrain = load_data('train')
    xvalid, yvalid = load_data('dev')
    stat_encode = np.load('stat_encode.npy')
    projection_bias = torch.FloatTensor(unigram_logits(stat_encode))
    train_loader = DataLoader(xtrain, ytrain, batch_size=args.batch_size)
    valid_loader = DataLoader(xvalid, yvalid, batch_size=args.batch_size)
    if args.model:
        print('loading model: {}'.format(args.model))
        model = torch.load(args.model)
    else:
        model = LASModel(projection_bias=projection_bias)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = CrossEntropyLoss3D(reduce=False)

    if CUDA_AVAILABLE:
        model.cuda()
        criterion.cuda()

    min_loss = 300.
    for epoch in range(args.epoch):
        print('-' * 10 + 'epoch {}: train'.format(epoch) + '-' * 10)
        loss = routine(args, model, train_loader, optimizer, criterion, epoch, True)
        print('train epoch {}:\tloss={:.4f}'.format(epoch, loss))
        print('-' * 10 + 'epoch {}: evaluate'.format(epoch) + '-' * 10)
        loss = routine(args, model, valid_loader, optimizer, criterion, epoch, False)
        print('valid epoch {}:\tloss={:.4f}'.format(epoch, loss))
        if loss < min_loss:
            torch.save(model, '{}/{:.4f}.pt'.format(args.save_dir, loss))
        min_loss = min(loss, min_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=16)
    parser.add_argument('--model', '-m', dest='model', type=str, default=None)
    parser.add_argument('--print-freq', dest='print_freq', type=int, default=10)
    parser.add_argument('--epoch', dest='epoch', type=int, default=30)
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='models')
    args = parser.parse_args()
    print(args)
    main(args)