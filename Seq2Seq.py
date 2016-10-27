import random
import numpy as np
import mxnet as mx
from lstm import lstm_unroll, lstm_without_softmax
from data import Seq2SeqIter, default_build_vocab
from tqdm import tqdm
import logging


class Seq2Seq(object):
    def __init__( self, seq_len, batch_size, num_layers,
                  input_size, embed_size, hidden_size,
                  output_size, dropout, mx_ctx=mx.cpu() ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size = output_size
        self.mx_ctx = mx_ctx
        self.initializer = mx.initializer.Uniform(0.1)
        self.params_blocks = []
        self.enc_lstm_exe = self.build_lstm_encoder()
        self.dec_lstm_exe = self.build_lstm_decoder()

        opt = mx.optimizer.create('rmsprop')
        opt.lr = 0.05
        self.updater = mx.optimizer.get_updater(opt)

    def build_lstm_encoder( self ):
        enc_lstm = lstm_without_softmax(num_lstm_layer=self.num_layers, seq_len=self.seq_len,
                                        input_size=self.input_size, num_hidden=self.hidden_size,
                                        num_embed=self.embed_size, dropout=0.)
        init_c = [('l%d_init_c' % l, (self.batch_size, self.hidden_size)) for l in range(self.num_layers)]
        init_h = [('l%d_init_h' % l, (self.batch_size, self.hidden_size)) for l in range(self.num_layers)]
        init_states = init_c + init_h
        input_data = [('data', (self.batch_size, self.seq_len))]
        provide_shape = dict(input_data + init_states)

        arg_shape, output_shape, aux_shape = enc_lstm.infer_shape(**provide_shape)
        arg_names = enc_lstm.list_arguments()

        args = {}
        args_grad = {}
        grad_req = {}
        for name, shape in dict(zip(arg_names, arg_shape)).items():
            args[name] = mx.nd.zeros(shape, self.mx_ctx)
            if name in ['data'] or name.endswith('init_c') or name.endswith('init_h'):
                continue
            self.initializer(name, args[name])
            args_grad[name] = mx.nd.zeros(shape, self.mx_ctx)
            grad_req[name] = 'write'

        enc_lstm_exe = enc_lstm.bind(ctx=self.mx_ctx, args=args, args_grad=args_grad, grad_req=grad_req)

        for name in args_grad.keys():
            self.params_blocks.append((len(self.params_blocks) + 1, name, args[name], args_grad[name]))

        return enc_lstm_exe

    def build_lstm_decoder( self ):
        dec_lstm = lstm_unroll(num_lstm_layer=self.num_layers, seq_len=self.seq_len,
                               input_size=self.input_size, num_hidden=self.hidden_size,
                               num_embed=self.embed_size, num_label=self.output_size)

        init_c = [('l%d_init_c' % l, (self.batch_size, self.hidden_size)) for l in range(self.num_layers)]
        init_h = [('l%d_init_h' % l, (self.batch_size, self.hidden_size)) for l in range(self.num_layers)]
        init_states = init_c + init_h
        input_data = [('data', (self.batch_size, self.seq_len))]
        provide_data = input_data + init_states
        provide_label = [('softmax_label', (self.batch_size, self.seq_len))]
        provide_args = dict(provide_data + provide_label)
        arg_shape, output_shape, _ = dec_lstm.infer_shape(**provide_args)
        arg_names = dec_lstm.list_arguments()

        args = {}
        args_grad = {}
        grad_req = {}
        for shape, name in zip(arg_shape, arg_names):
            args[name] = mx.nd.zeros(shape, self.mx_ctx)
            if name in ['softmax_label', 'data'] or name.endswith('init_c'):
                continue
            args_grad[name] = mx.nd.zeros(shape, self.mx_ctx)
            grad_req[name] = 'write'

        for name in arg_names:
            if name in ['data', 'softmax_label'] or \
                    name.endswith('init_h') or name.endswith('init_c'):
                continue
            self.initializer(name, args_grad[name])

        dec_lstm_exe = dec_lstm.bind(ctx=self.mx_ctx, args=args,
                                     args_grad=args_grad, grad_req=grad_req)

        for name in args_grad.keys():
            self.params_blocks.append((len(self.params_blocks) + 1, name, args[name], args_grad[name]))
        print self.params_blocks

        return dec_lstm_exe

    def forward_connect( self, enc_out ):
        for l in range(self.num_layers):
            enc_out.copyto(self.dec_lstm_exe.arg_dict['l%d_init_h' % l])

    def get_params( self ):
        for i, name in enumerate(arg_names):
            if name in ['softmax_label', 'data']:  # input, output
                continue
        initializer(name, arg_dict[name])

        param_blocks.append((i, arg_dict[name], args_grad[name], name))

    def train( self, data ):
        for iter in tqdm(data):
            enc_batch_in = iter['enc_batch_in']
            dec_batch_in = iter['dec_batch_in']
            dec_batch_tr = iter['dec_batch_tr']
            self.enc_lstm_exe.arg_dict['data'] = enc_batch_in
            batch_rep = self.enc_lstm_exe.forward(is_train=True)[0]
            self.forward_connect(batch_rep)
            self.dec_lstm_exe.arg_dict['data'] = dec_batch_in
            self.dec_lstm_exe.arg_dict['softmax_label'] = dec_batch_tr

            # the output of dec_lstm_exe.forward is [batch_size * seq_len, vocab_size]
            self.dec_lstm_exe.forward(is_train=True)
            # print dec_batch_in.shape
            # print self.dec_lstm_exe.outputs[0].shape
            self.dec_lstm_exe.backward()

            grad_out = self.dec_lstm_exe.grad_arrays[4]
            self.enc_lstm_exe.backward([grad_out])

            for i, name, params, grad_params in self.params_blocks:
                self.updater(i, grad_params, params)


if __name__ == '__main__':
    vocab, vocab_rsd = default_build_vocab('./data/vocab.txt')
    print 'vocab size is %d' % len(vocab)
    data = Seq2SeqIter(data_path='data.pickle', source_path='./data/a.txt',
                       target_path='./data/b.txt', vocab=vocab,
                       vocab_rsd=vocab_rsd, batch_size=36, max_len=25,
                       data_name='data', label_name='label', split_char='\n',
                       text2id=None, read_content=None, model_parallel=False)
    print 'data size is %d' % data.size
    model = Seq2Seq(seq_len=25, batch_size=36, num_layers=1,
                    input_size=len(vocab), embed_size=150, hidden_size=150,
                    output_size=len(vocab), dropout=0.0, mx_ctx=mx.cpu())
    model.train(data)
