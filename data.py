# coding=utf-8

import sys, nltk
import numpy as np
import mxnet as mx
import h5py, pickle


def default_read_content( path ):
    with open(path) as ins:
        content = ins.read()
        return content


def default_build_vocab( vocab_path ):
    vocab = default_read_content(vocab_path)
    vocab = vocab.split('\n')
    idx = 1
    vocab_rsd = {}
    for word in vocab:
        vocab_rsd[word] = idx
        idx += 1
    return vocab, vocab_rsd


def default_gen_buckets( sentences, batch_size, the_vocab ):
    len_dict = {}
    max_len = -1
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1
    # print(len_dict)

    tl = 0
    buckets = []
    for l, n in len_dict.items():
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


def default_text2id( sentence, the_vocab, max_len, vocab ):
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    tokens = []
    for w in words:
        if len(w) == 0: continue
        if len(tokens) >= max_len: break
        if not w in vocab:
            tokens.append(the_vocab['<unknown>'])
        else:
            tokens.append(the_vocab[w])
    return tokens


def default_gen_buckets( len_dict, batch_size ):
    tl = 0
    buckets = []
    for l, n in len_dict.items():
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    return buckets


class Seq2SeqIter(mx.io.DataIter):
    def __init__( self, data_path, source_path, target_path, vocab, vocab_rsd, batch_size,
                  max_len, data_name='data', label_name='label', split_char='\n',
                  text2id=None, read_content=None, model_parallel=False, ctx=mx.cpu()):
        super(Seq2SeqIter, self).__init__()

        self.ctx = ctx
        self.iter_data = []
        if data_path is not None:
            print 'loading data set'
            with open(data_path, 'r') as f:
                self.iter_data = pickle.load(f)
                self.size = len(self.iter_data)
        self.vocab = vocab
        self.vocab_rsd = vocab_rsd
        self.vocab_size = len(vocab)
        self.data_name = data_name
        self.label_name = label_name
        self.model_parallel = model_parallel
        self.batch_size = batch_size
        self.max_len = max_len
        self.source_path = source_path
        self.target_path = target_path
        self.split_char = split_char

        if text2id is None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content is None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content

        if len(self.iter_data) == 0:
            self.iter_data = self.make_data_iter_plan()

    def make_data_iter_plan( self ):
        source = self.read_content(self.source_path)
        source_lines = source.split(self.split_char)

        target = self.read_content(self.target_path)
        target_lines = target.split(self.split_char)

        self.size = len(source_lines)
        self.suffer_ids = np.random.permutation(self.size)

        self.enc_inputs = []
        self.dec_inputs = []
        self.dec_targets = []
        len_dict = {}
        cnt = 0
        for i in range(self.size):
            source = source_lines[i]
            target = target_lines[i]
            t1 = source.split('\t\t')
            t2 = target.split('\t\t')
            if len(t1) < 2 or len(t2) < 2: continue
            st, su = t1[0], t1[1]
            tt, tu = t2[0], t2[1]
            dec_input = []
            dec_target = []
            s_tokens = self.text2id(st, vocab_rsd, self.max_len, vocab)
            t_tokens = self.text2id(tt, vocab_rsd, self.max_len, vocab)
            self.enc_inputs.append(s_tokens)
            dec_input.append(self.vocab_rsd['<go>'])
            dec_input[1:len(t_tokens) + 1] = t_tokens[:]
            self.dec_inputs.append(dec_input)
            dec_target[:len(t_tokens)] = t_tokens[:]
            dec_target.append(self.vocab_rsd['<eos>'])
            self.dec_targets.append(dec_target)
            if len(dec_input) < 3: continue
            if not len(dec_input) in len_dict.keys():
                len_dict[len(dec_input)] = 1
            else:
                len_dict[len(dec_input)] += 1
            cnt += 1
        self.buckets = default_gen_buckets(len_dict, self.batch_size)
        self.len_dict = len_dict
        self.size = cnt
        bucket_n_batches = {}
        for l, n in self.len_dict.items():
            if l < 3:
                continue
            bucket_n_batches[l] = n / self.batch_size
        # print bucket_n_batches

        data_buffer = {}
        for i in range(self.size):
            dec_input = self.dec_inputs[i]
            if len(dec_input) < 3:
                continue
            enc_input = self.enc_inputs[i]
            dec_target = self.dec_targets[i]
            if not len(dec_input) in data_buffer.keys():
                data_buffer[len(dec_input)] = []
                data_buffer[len(dec_input)].append({
                    'enc_input': enc_input,
                    'dec_input': dec_input,
                    'dec_target': dec_target
                })
            else:
                data_buffer[len(dec_input)].append({
                    'enc_input': enc_input,
                    'dec_input': dec_input,
                    'dec_target': dec_target
                })

        iter_data = []
        for l, n in self.len_dict.items():
            for k in range(0, n, self.batch_size):
                if k + self.batch_size >= self.size: break
                encin_batch = np.zeros((self.batch_size, self.max_len))
                decin_batch = np.zeros((self.batch_size, l))
                dectr_batch = np.zeros((self.batch_size, l))
                if n < self.batch_size: break
                for j in range(self.batch_size):
                    one = data_buffer[l][j]
                    encin = one['enc_input']
                    offset = self.max_len - len(encin)
                    encin_batch[j][offset:] = encin
                    decin_batch[j] = one['dec_input']
                    dectr_batch[j] = one['dec_target']

                iter_data.append({
                    'enc_batch_in': encin_batch,
                    'dec_batch_in': decin_batch,
                    'dec_batch_tr': dectr_batch
                })
        with open('data.pickle', 'w') as f:
            print 'dumping data'
            pickle.dump(iter_data, f)
        return iter_data

    def __iter__( self ):
        for batch in self.iter_data:
            yield batch


class SimpleBatch(object):
    def __init__( self, data_names, data, label_names, label, bucket_key ):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None

    @property
    def provide_data( self ):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label( self ):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


if __name__ == '__main__':
    vocab, vocab_rsd = default_build_vocab('./data/vocab.txt')
    data = Seq2SeqIter(data_path=None,source_path='./data/a.txt', target_path='./data/b.txt',
                       vocab=vocab, vocab_rsd=vocab_rsd, batch_size=36, max_len=25,
                       data_name='data', label_name='label', split_char='\n',
                       text2id=None, read_content=None, model_parallel=False)
    for iter in data:
        print iter['enc_batch_in'].shape
