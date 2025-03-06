import torch.nn as nn

import torch
class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # print('encoded_spconv_tensor',encoded_spconv_tensor.shape)
        spatial_features = encoded_spconv_tensor.dense()
        # print(spatial_features.shape)
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
class HeightCompression_xishugaoduyas(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # print('encoded_spconv_tensor',encoded_spconv_tensor.shape)
        spatial_features = encoded_spconv_tensor.dense()
        # print(spatial_features.shape)
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class HeightCompression_xxx(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # # print('encoded_spconv_tensor',encoded_spconv_tensor.shape)
   
   
        # print(encoded_spconv_tensor.features.shape)
        # print(encoded_spconv_tensor.indices.shape)
        # print(batch_dict['encoded_spconv_tensor'].spatial_shape)
        # print(batch_dict['encoded_spconv_tensor'].batch_size)

        # N, C, D, H, W = spatial_features.shape
        # print(spatial_features.shape)
        # # spatial_features = spatial_features.view(N, C * D, H, W)
        # spatial_features = spatial_features.view(N * C * D * H * W, 1)
        


        # batch_dict['spatial_features'] = spatial_features

        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        def forward(self, batch_dict):
            """
                Args:
                    batch_dict:
                        encoded_spconv_tensor: sparse tensor
                Returns:
                    batch_dict:
                        spatial_features:
            """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        
        # 打印出 encoded_spconv_tensor 的特征和索引维度
        print(encoded_spconv_tensor.features.shape)
        print(encoded_spconv_tensor.indices.shape)
        print(batch_dict['encoded_spconv_tensor'].spatial_shape)
        print(batch_dict['encoded_spconv_tensor'].batch_size)
        
        # # 转换为密集张量
        # spatial_features = encoded_spconv_tensor.dense()
        # N, C, D, H, W = spatial_features.shape
        
        # # 将深度维度 D 和高度维度 H 压缩
        # spatial_features = spatial_features.view(N, C * D, H, W)  # 合并深度维度和高度维度
        
        # # 修改索引：去掉深度维度，保留 batch_index, y, x
        # indices_cat = encoded_spconv_tensor.indices[:, [0, 2, 3]]  # 保留 batch_index, y, x
        # # 计算唯一索引
        # indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        
        # # 创建唯一索引的特征，并对重复的索引进行累加
        # features_unique = encoded_spconv_tensor.features.new_zeros((indices_unique.shape[0], encoded_spconv_tensor.features.shape[1]))
        # features_unique.index_add_(0, _inv, encoded_spconv_tensor.features)
        
        # # 更新 batch_dict 并返回
        # batch_dict['spatial_features'] = spatial_features
        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # batch_dict['encoded_spconv_tensor'] = spconv.SparseConvTensor(
        #     features=features_unique,
        #     indices=indices_unique,
        #     spatial_shape=batch_dict['encoded_spconv_tensor'].spatial_shape[1:],  # 新的空间形状
        #     batch_size=batch_dict['encoded_spconv_tensor'].batch_size
        # )
        # print(batch_dict['encoded_spconv_tensor'].features.shape)
        # print(encoded_spconv_tensor.indices.shape)
        # print(batch_dict['encoded_spconv_tensor'].spatial_shape)
        # print(batch_dict['encoded_spconv_tensor'].batch_size)

        return batch_dict

        # return batch_dict