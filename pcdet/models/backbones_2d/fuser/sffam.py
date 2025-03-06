import torch
from torch import nn
from ....utils.spconv_utils import replace_feature, spconv
import numpy as np
from functools import partial
import json
from functools import partial
# from spconv.pool import SparseMaxPool2d, SparseMaxPool3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

import cv2
import numpy as np



class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.Linear1 = nn.Sequential(nn.Linear(512,256,bias=False),
                                       nn.ReLU())

        self.Linear2 = nn.Sequential(nn.Linear(256, 256,bias=False),
                                       nn.ReLU())

        self.Linear3 = nn.Sequential(nn.Linear(256,256,bias=False),
                                       )
                                       

    def forward(self, matched_features1 , matched_features2):
        
        print('matched_features1_im',matched_features1.shape)
        print('matched_features2_li',matched_features2.shape)

        cat_features = torch.cat([matched_features1, matched_features2], dim=-1)
        print('cat_features',cat_features.shape)

        features1 = self.Linear1(cat_features)
        
        features2 = self.Linear2(features1)
        features3 = self.Linear3(features2)

        features_attention = torch.sigmoid(features3)
        
        features1_1 = features1 * features_attention

        fused_features = features1_1 + matched_features2
        return fused_features



def replace_sparse_features(sparse_tensor, fused_tensor_features, matching_idx):
    # 获取稀疏张量的特征点数量
    num_points = sparse_tensor.indices.shape[0]

    # 初始化新的稀疏特征张量，大小为 (num_points, fused_tensor_features.shape[1])
    new_sparse_features = torch.zeros((num_points, fused_tensor_features.shape[1]), device=fused_tensor_features.device)

    # 使用 matching_idx 进行特征替换
    new_sparse_features[matching_idx[:, 1]] = fused_tensor_features

    # 创建新的 SparseConvTensor，使用新替换的特征，并保留原始的坐标、形状等元信息
    new_sparse_tensor = spconv.SparseConvTensor(
        features=new_sparse_features,
        indices=sparse_tensor.indices,  # 保留原稀疏张量的坐标信息
        spatial_shape=sparse_tensor.spatial_shape,  # 保持原始空间形状
        batch_size=sparse_tensor.batch_size
    )

    return new_sparse_tensor
def match_coords_with_hash(coords1, coords2):
    # 构建哈希表，将 coords2 中的坐标存入字典
    coords2_dict = {tuple(c.tolist()): i for i, c in enumerate(coords2)}
    
    # 初始化匹配索引的列表
    matching_idx = []
    
    # 遍历 coords1 中的每个坐标，查找是否在哈希表中
    for i, c1 in enumerate(coords1):
        c1_tuple = tuple(c1.tolist())  # 转换为 tuple 以便哈希查找
        if c1_tuple in coords2_dict:
            matching_idx.append([i, coords2_dict[c1_tuple]])
    
    # 转换为 Tensor 返回
    if matching_idx:
        return torch.tensor(matching_idx)
    else:
        return torch.empty((0, 2), dtype=torch.long)

def match_and_fuse_sparse_xin(sparse_tensor1, sparse_tensor2):
    # 获取两个SparseConvTensor的特征和坐标
    features1, coords1 = sparse_tensor1.features, sparse_tensor1.indices
    features2, coords2 = sparse_tensor2.features, sparse_tensor2.indices


    matching_idx = match_coords_with_hash(coords1, coords2)

    # 提取匹配的特征
    matched_features1 = features1[matching_idx[:, 0]]
    matched_features2 = features2[matching_idx[:, 1]]


    
    matched_coords1 = coords1[matching_idx[:, 0]]
    matched_coords2 = coords2[matching_idx[:, 1]]

    # 假设你希望保留sparse_tensor1和sparse_tensor2的匹配索引，可以选择任意一个作为最终索引
    fused_indices = matched_coords2  # 或者用 matched_coords2
   


    return  matched_features1 , matched_features2, fused_indices

class SFFAM(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.attention=Attention()
        
        
    def transform_to_sparse(self,x):

        x = x.permute(0,2,3,1).contiguous()
        x = spconv.SparseConvTensor.from_dense(x)
        return x

    
    def forward(self, batch_dict):
        # img_bev = batch_dict['image_fpn'][0]  # 图像特征
        img_bev = batch_dict['spatial_features_img']
        # print(img_bev.shape)
        print('img_bev', img_bev.shape)

        img_bev = self.transform_to_sparse(img_bev)  # 转换为稀疏卷积格式
        # print(img_bev)
        # print(img_bev.indices.shape)
        print( img_bev)

        print('img_bev_sp', img_bev.features.shape)
        print(img_bev.indices.shape)

        lidar_bev = batch_dict['encoded_spconv_tensor']  # 雷达特征
        # print(lidar_bev)
        # print(lidar_bev.indices.shape)

        matched_features1, matched_features2 ,fused_indices = match_and_fuse_sparse_xin(img_bev, lidar_bev)
        fused_features = self.attention(matched_features1, matched_features2)
        


        fused_features_sparse_tensor = spconv.SparseConvTensor(
            features=fused_features,           # 使用提取后的非零元素的特征
            indices=fused_indices,                # 根据非零元素重新构建的索引
            spatial_shape=lidar_bev.spatial_shape,  # 保持空间形状一致
            batch_size=lidar_bev.batch_size         # 保持原始batch size
        )


        batch_dict.update({
            'encoded_spconv_tensor': fused_features_sparse_tensor,
        })

        return batch_dict


