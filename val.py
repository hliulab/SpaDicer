import numpy as np
import pandas as pd
import torch.utils.data
from scipy.stats import pearsonr
import torch.nn.functional as F
import config
import utils
from dataset import ValDataset, ValDataset_cls
from model.SpaDicer import SpaDicer
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
print("data loader...")
target_set = ValDataset(data_path=config.sc_data,
                        label_path=config.sc_location,
                        dataset_type='target')

source_set = ValDataset(data_path=config.st_data,
                        label_path=config.st_cell_rations,
                        dataset_type='source')
source_celltype_set = ValDataset_cls(data_path=config.st_data,
                                    label_path=config.st_major_celltype)

dataloader_source = torch.utils.data.DataLoader(
    dataset=source_set,
    batch_size=config.val_st_batch,
    shuffle=True,
    num_workers=1
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=target_set,
    batch_size=config.val_sc_batch,
    shuffle=True,
    num_workers=1
)
dataloader_source_celltype = torch.utils.data.DataLoader(
    dataset=source_celltype_set,
    batch_size=config.val_st_batch,
    shuffle=True,
    num_workers=1
)
print("data loader finished!")

def val():
    model = SpaDicer(config.inf, config.input_class).to(config.device)
    checkpoint = torch.load(config.model_path, map_location=torch.device(config.device))
    model.load_state_dict(checkpoint)
    model.eval()
    all_pearson_target = []
    all_pearson_x = []
    all_pearson_y = []
    all_pred_x = []
    all_pred_y = []
    all_label_x = []
    all_label_y = []
    for data, label in dataloader_target:
        data = data.to(config.device)
        label = label.to(config.device)
        result = model(input_data=data, mode='source', rec_scheme='all')
        _, _, pred, _ = result
        pred_np = pred.cpu().detach().numpy()
        label_np = label.cpu().detach().numpy()
        pred_dist = utils.cal_dist(pred_np)
        label_dist = utils.cal_dist(label_np)
        pearson_dist, _ = pearsonr(pred_dist, label_dist)
        all_pearson_target.append(pearson_dist)
        pearson_x, _ = pearsonr(pred_np[:, 0], label_np[:, 0])
        pearson_y, _ = pearsonr(pred_np[:, 1], label_np[:, 1])
        all_pearson_x.append(pearson_x)
        all_pearson_y.append(pearson_y)

        all_pred_x.extend(pred_np[:, 0])
        all_pred_y.extend(pred_np[:, 1])
        all_label_x.extend(label_np[:, 0])
        all_label_y.extend(label_np[:, 1])
    original_df = pd.read_csv(config.sc_location)
    original_df['Pred_X'] = np.nan
    original_df['Pred_Y'] = np.nan
    for pred_x, pred_y, label_x, label_y in zip(all_pred_x, all_pred_y, all_label_x, all_label_y):
        mask = (original_df['X'] == label_x) & (original_df['Y'] == label_y)
        original_df.loc[mask, 'Pred_X'] = pred_x
        original_df.loc[mask, 'Pred_Y'] = pred_y
    celltype = pd.read_csv(config.sc_meta)
    original_df['celltype'] = celltype['celltype']


    '''Verify spatial transcriptomics single-cell type deconvolution'''
    all_pearson_source = []
    all_ssim_source = []
    all_rmse_source = []
    for data, label in dataloader_source:
        data = data.to(config.device)
        label = label.to(config.device)
        result = model(input_data=data, mode='target', rec_scheme='all')
        _, _, pred, _ = result
        pred = F.softmax(pred, dim=1)
        for i in range(pred.shape[0]):
            pred_np = pred[i].cpu().detach().numpy()
            label_np = label[i].cpu().detach().numpy()
            pearson, _ = pearsonr(pred_np, label_np)
            all_pearson_source.append(pearson)
            # 计算SSIM
            if pred_np.shape == label_np.shape:
                data_range = pred_np.max() - pred_np.min()
                ssim_index = ssim(pred_np, label_np, data_range=data_range)
                all_ssim_source.append(ssim_index)
            # 计算RMSE
            rmse = mean_squared_error(label_np, pred_np, squared=False)
            all_rmse_source.append(rmse)

    '''Verify the major cell types in spatial transcriptomics single-cell type deconvolution'''
    all_label = []
    all_pred = []
    for data, label in dataloader_source_celltype:
        data = data.to(config.device)
        label = label.to(config.device)
        result = model(input_data=data, mode='target', rec_scheme='all')
        _, _, pred, _ = result
        pred = F.softmax(pred, dim=1)
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        all_pred.extend(pred)
        all_label.extend(label)

    auc = utils.cal_roc_auc_score(np.array(all_label), np.array(all_pred), config.input_class)
    utils.plot_auc(config.label_dict, all_label, all_pred, config.result_path)
    return np.mean(np.array(all_pearson_target)), np.mean(np.array(all_pearson_x)), np.mean(np.array(all_pearson_y)), np.mean(np.array(all_pearson_source)), auc
