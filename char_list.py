ENCODE_MAP = {'%': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

DECODE_MAP = ['%', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
assert(len(ENCODE_MAP) == len(DECODE_MAP))
CHAR_SIZE = len(ENCODE_MAP)

SOS = '%'
EOS = '%'
SOS_IDX = ENCODE_MAP[SOS]
EOS_IDX = ENCODE_MAP[EOS]
