import h5py
import pickle
import pprint
import mxnet as mx
from lstm_test import lstm,lstm_unroll, lstm_unroll_wo_emb

lstm = lstm_unroll_wo_emb(num_lstm_layer=1, seq_len=4, num_hidden=20, num_label=5, dropout=0.)
print lstm

init_c = [('l0_init_c', (3, 20))]
init_h = [('l0_init_h', (3, 20))]
init_states = init_c + init_h

provide_data = [('data', (3, 4, 10))] + init_states
provide_label = [('softmax_label', (3,4))]
provide = dict(provide_label + provide_data)
# arg_shape, output_shape, _ = lstm.infer_shape(**provide)
# print arg_shape
# print output_shape
# arg_names = lstm.list_arguments()
# arg_arrays = [mx.nd.zeros(s) for s in arg_shape]
# args_grad = {}
# print arg_names
# for shape, name in zip(arg_shape, arg_names):
#     if name in ['softmax_label', 'data']:
#         continue
#     args_grad[name] = mx.nd.zeros(shape)

exe = lstm.simple_bind(ctx=mx.cpu(), **provide)

data = mx.sym.Variable('data')
embed_weight = mx.sym.Variable("embed_weight")
embed = mx.sym.Embedding(data=data, input_dim=5,
                         weight=embed_weight, output_dim=10, name='embed')

emd_shape = dict([('data', (3, 4))])

emb_exe = embed.simple_bind(ctx=mx.cpu(), **emd_shape)

seq = mx.nd.ones((3,4))
seq.copyto(emb_exe.arg_dict['data'])
res = emb_exe.forward()[0]
print res.shape

data = mx.nd.ones((3, 4, 10))
h = mx.nd.zeros((3, 20))
c = mx.nd.zeros((3, 20))

data.copyto(exe.arg_dict['data'])
h.copyto(exe.arg_dict['l0_init_h'])
c.copyto(exe.arg_dict['l0_init_c'])

res = exe.forward(is_train=True)[0]
print res.shape

