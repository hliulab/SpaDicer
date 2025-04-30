import random
import torch
from torch.backends import cudnn

label_dict = {
	0: 0
    , 1: 1
    , 2: 2
    , 3: 3
    , 4: 4, 5: 5,  6: 6, 7: 7
}
input_class = 8
inf = 610
model_root = 'model'
cuda = True
cudnn.benchmark = True
lr = 1e-3
n_epoch = 200
batchsize=64
step_decay_weight = 0.95
lr_decay_step = 20000
weight_decay = 1e-6
momentum = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
'''Gene expression profiles from single-cell and spatial transcriptomics need to be filtered for highly variable genes using preprocessing.
    sc_data and st_data
'''
sc_data = r''
sc_meta = r''
st_data = r''
st_meta = r''
sc_location = r''
st_cell_rations = r''
st_major_celltype = r''
val_sc_batch = 1523
val_st_batch = 189
model_path = r''
result_path = r''
sc_rawdata = r''
st_rawdata = r''