import os
import sys
from collections import Counter

#make vocab
data_dir = 'data/train'
vocab_dir = 'data/vocab'
for name in ['source', 'target']:
    with open(os.path.join(data_dir, '{}.txt'.format(name))) as f:
        words = []
        for text in f.readlines():
            for token in text.split():
                words.append(token)
        count = Counter(words)
        print(len(count))

        vocab = {}
        vocab['<pad>'] = len(vocab)
        vocab['<unk>'] = len(vocab)

        w = open(os.path.join(vocab_dir, '{}.txt'.format(name)), 'w')
        w.write('<pad>\n')
        w.write('<unk>\n')
        for word, index in count.most_common():
            if index > 3:
                w.write(word+'\n')
                vocab[word] = len(vocab)
        w.close()