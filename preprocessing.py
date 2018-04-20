import argparse
import numpy as np


def encode(args, array, char_map):
    output = []
    for s in array:
        arr = np.zeros(len(s) + 2, dtype=int)
        arr[0] = char_map[args.SOS]
        arr[1: -1] = [char_map[c] for c in s]
        arr[-1] = char_map[args.EOS]
        # for i, c in enumerate(s):
        #     arr[i + 1] = char_map[c]
        output.append(arr)
    return np.array(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', dest='SOS', type=str, default='%')
    parser.add_argument('--end', '-e', dest='EOS', type=str, default='%')
    args = parser.parse_args()
    print(args)

    SOS = args.SOS
    EOS = args.EOS

    dev = np.load('dev_transcripts.npy')
    train = np.load('train_transcripts.npy')
    trans = np.concatenate((dev, train))

    count = dict()
    for s in trans:
        for c in s:
            if c not in count:
                count[c] = 0
            count[c] += 1

    stat = list(count.items())
    stat.sort(key=lambda x: x[0])

    # add SOS and EOS
    count[SOS] = count[EOS] = len(trans)
    if EOS != SOS:
        stat = [(SOS, len(trans))] + stat + [(EOS, len(trans))]
    else:
        stat = [(SOS, 2 * len(trans))] + stat

    char_map = dict()
    index_map = []
    for i, (key, _) in enumerate(stat):
        char_map[key] = i
        index_map.append(key)

    print(char_map)
    print(index_map)

    dev_encode = encode(args, dev, char_map)
    train_encode = encode(args, train, char_map)

    np.save('dev_encode.npy', dev_encode)
    np.save('train_encode.npy', train_encode)

    stat_encode = np.array([[char_map[k], v] for k, v in stat])
    np.save('stat_encode.npy', stat_encode)


if __name__ == '__main__':
    main()