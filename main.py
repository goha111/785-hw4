from model import *
from char_list import *

import argparse
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
    for i, data in enumerate(loader):
        seq, seq_len, label_in, label_out, label_len, label_mask = to_cuda(*data)
        seq_len = seq_len.data.cpu().numpy()   # for pack_padded_sequence
        label_in = label_in.long()
        logits = model(seq, seq_len, label_in, label_len)

    if train:
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
    return logits


def main(args):
    xtrain, ytrain = load_data('dev')
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

    for epoch in range(args.epoch):
        print('-' * 10 + 'epoch {}: train'.format(epoch) + '-' * 10)
        loss, _ = routine(args, model, train_loader, optimizer, criterion, epoch, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()
    main(args)