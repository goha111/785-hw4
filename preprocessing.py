import numpy as np

SOS = '<'
EOS = '>'

def encode(array, char_map):
    output = []
    for s in array:
        arr = np.zeros(len(s) + 2, dtype=int)
        arr[0] = char_map[SOS]
        arr[-1] = char_map[EOS]
        for i, c in enumerate(s):
            arr[i + 1] = char_map[c]
        output.append(arr)
    return np.array(output)

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
stat = [(SOS, len(trans))] + stat + [(EOS, len(trans))]

char_map = dict()
index_map = []
for i, (key, _) in enumerate(stat):
    char_map[key] = i
    index_map.append(key)

print(char_map)
print(index_map)

dev_encode = encode(dev, char_map)
train_encode = encode(train, char_map)

np.save('dev_encode.npy', dev_encode)
np.save('train_encode.npy', train_encode)

stat_encode = np.array([[char_map[k], v] for k, v in stat])
np.save('stat_encode.npy', stat_encode)