import torch 
import argparse
import numpy as np
from data_provider import datasets_factory
from utils import preprocess, metrics
from layers import CausalLSTMStack  # noqa

parser = argparse.ArgumentParser(description='Process some integers.')
args = parser.parse_args()
args.dataset_name = "mnist"
args.train_data_paths = 'data/moving-mnist-example/moving-mnist-train.npz'
args.valid_data_paths = 'data/moving-mnist-example/moving-mnist-valid.npz'
args.save_dir = 'checkpoints/mnist_predrnn_pp'
args.img_width  = 64
args.batch_size = 8
args.patch_size = 1
args.seq_length = 19
args.num_hidden = [128, 64, 64, 64]
args.num_layers = len(args.num_hidden)
args.lr = 0.0001

##### load the train data
train_input_handle, test_input_handle = datasets_factory.data_provider(
    args.dataset_name, args.train_data_paths, args.valid_data_paths,
    args.batch_size, args.img_width)

#  tmp = np.load(args.train_data_paths)
#  tmp = tmp['input_raw_data']
#  print(tmp.shape)
#  print(tmp[0].shape)
#print(type(train_input_handle), type(test_input_handle))

model  = CausalLSTMStack(3, 2, args.num_hidden) #filter_size, num_dims
decoder = torch.nn.Conv2d(1, 1, 1, 1)
# tmp = np.random.rand(8, 20, 16, 16, 16)


###  run iters

model.cuda()
decoder.cuda()
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()), lr=args.lr)

for itr in range(10000):
    ims = train_input_handle.get_batch()
    ims = preprocess.reshape_patch(ims, args.patch_size)
    #print(ims.shape)# (8, 20, 16, 16, 16)
    ims = np.swapaxes(ims, 0, 1)
    h, c, m, z = [None]*4
    #print(ims.shape)# (20, 8, 16, 16, 16)
    for t in range(args.seq_length):
        tmp = torch.Tensor(ims[t])
        tmp = tmp.cuda()
        h, c, m, z = model(tmp, h, c, m, z)

    z = decoder(h[-1].permute(0,-1,1,2)).permute(0,2,3,1)
    y = torch.Tensor(ims[-1])
    y = y.cuda()
    loss = loss_fn( z, y )
    loss.backward()
    optim.step()

    #print(len(h), len(c), len(m), len(z))
    #print("h", h[-1].shape)
    print("loss = ", loss.item())



