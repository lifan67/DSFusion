import torch
import torch.nn as nn
import torch.nn.functional as F
# from .pinet import PCAttention
# from .vfe_template import VFETemplate

class PCAttention(nn.Module):
    def __init__(self, gate_channels, reduction_rate, pool_types=['max', 'mean'], activation=nn.ReLU(),
                 ):
        super(PCAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_rate),
            activation,
            nn.Linear(gate_channels // reduction_rate, gate_channels)
        )
        self.pool_types = pool_types
        print('self.pool_types', self.pool_types)

        self.max_pool = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None)) #if not channel_mean else GetChannelMean(keepdim=True)
        print('self.max_pool, self.avg_pool',self.max_pool, self.avg_pool)
    def forward(self, x):
        '''
        # shape [n_voxels, channels, n_points] for point-wise attention
        # shape [n_voxels, n_points, channels] for channels-wise attention
        '''
        attention_sum = None
        for pool_type in self.pool_types:
            # [n_voxels, 1, n_points]
            if pool_type == 'max':
                max_pool = self.max_pool(x)
                attention_raw = self.mlp(max_pool)
            elif pool_type == 'mean':
                avg_pool = self.avg_pool(x)
                attention_raw = self.mlp(avg_pool)
            if attention_sum is None:
                attention_sum = attention_raw
            else:
                attention_sum += attention_raw
        # scale = torch.sigmoid(attention_sum).permute(0, 2, 1)
        scale = attention_sum
        return scale

class Aten(nn.Module):
    def __init__(self, channel, out_channels):
        super(Aten, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(channel, out_channels, bias=False),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.fc(x)



        return y


class PFNLayer1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            self.avg=Aten(in_channels,out_channels)
            self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

            x_avg=self.avg(inputs)
            x_avg=x*x_avg

            x_sum=self.sum(inputs)
            x_sum=x*x_sum
        
        
        torch.backends.cudnn.enabled = False
        x_avg = self.norm(x_avg.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x_avg
        torch.backends.cudnn.enabled = True
        x_avg = F.relu(x_avg)
        # print(x_avg.shape)

        x_avg = torch.mean(x_avg, dim=1, keepdim=True)
        # print(x_avg.shape)
        

        torch.backends.cudnn.enabled = False
        x_sum = self.norm(x_sum.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x_sum
        torch.backends.cudnn.enabled = True
        x_sum = F.relu(x_sum)
        x_sum = torch.sum(x_sum, dim=1, keepdim=True)
        # print(x_sum.shape)
        
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        x_max=x_avg+(x_max+x_sum)/2


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        
class PFNLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            # self.avg=Aten(in_channels,out_channels)
            # self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            self.linear1 = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1, None)),
                nn.Linear(in_channels, out_channels //2, bias=False),
                nn.ReLU(),
                nn.Linear(out_channels //2, out_channels, bias=False),
                nn.Sigmoid()

            )
            self.norm1 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        # if inputs.shape[0] > self.part:
        #     # nn.Linear performs randomly when batch size is too large
        #     num_parts = inputs.shape[0] // self.part
        #     part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
        #                        for num_part in range(num_parts+1)]
        #     x = torch.cat(part_linear_out, dim=0)
        # else:
        x = self.linear(inputs)
            
        x_a = self.linear1(inputs)

        x_a=x*x_a


        torch.backends.cudnn.enabled = False
        x_a = self.norm1(x_a.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x_a
        torch.backends.cudnn.enabled = True
        x_a = F.relu(x_a)
        x_a = torch.mean(x_a, dim=1, keepdim=True)
        # print(x_a.shape)

        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        x_max=x_a+x_max


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PFNLayer3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            # self.avg=Aten(in_channels,out_channels)
            # self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)


        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

        self.point_att = PCAttention(32, reduction_rate=2, activation=nn.ReLU(),
                                         )
        
        self.channel_att = PCAttention(in_channels, reduction_rate=2, activation=nn.ReLU(),
                                           )
    def forward(self, inputs):

        x_p = self.point_att(inputs.permute(0,2,1)).permute(0, 2, 1)
        # print(x_p.shape)
        x_c = self.channel_att(inputs)
        # print(x_c.shape)
        x_a = torch.sigmoid(x_c*x_p)
        # print(x_a.shape)

        x_t=inputs*x_a






        x = self.linear(x_t)
            
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        # x_max=x_a+x_max


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated        

class PFNLayer4(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            # self.avg=Aten(in_channels,out_channels)
            # self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)

            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

        self.point_att = PCAttention(32, reduction_rate=4, activation=nn.ReLU(),
                                         )
        
        self.channel_att = PCAttention(in_channels, reduction_rate=4, activation=nn.ReLU(),
                                           )
    def forward(self, inputs):
        # print(inputs.shape)

        x_p = self.point_att(inputs.permute(0,2,1)).permute(0, 2, 1)
        # print(x_p.shape)
        x_c = self.channel_att(inputs)
        # print(x_c.shape)
        
        x_1 = inputs*torch.sigmoid(x_p)
        # print(torch.sigmoid(x_p).shape)
        # print(x_1.shape)

   

        x_2 = inputs*torch.sigmoid(x_c)
        # print(x_2.shape)

        x=x_1+x_2+inputs
        # print(x.shape)
        x=self.linear(inputs)


        # torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        # torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        # x_max=x_a+x_max


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated 
# if __name__ == '__main__':
    
#     x=torch.randn(22986, 32, 10)
#     # T=TModule()
#     # y=T(x)
#     # print(y.size())
#     T2=PFNLayer4(10,64)
#     y=T2(x)
#     print(y.size())


class PFNLayer4_point(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            # self.avg=Aten(in_channels,out_channels)
            # self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)

            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

        self.point_att = PCAttention(32, reduction_rate=4, activation=nn.ReLU(),
                                         )
        
        # self.channel_att = PCAttention(in_channels, reduction_rate=4, activation=nn.ReLU(),
        #                                    )
    def forward(self, inputs):

        x_p = self.point_att(inputs.permute(0,2,1)).permute(0, 2, 1)
        # print(x_p.shape)
        # x_c = self.channel_att(inputs)
        # print(x_c.shape)
        
        x_1 = inputs*torch.sigmoid(x_p)
        # print(torch.sigmoid(x_p).shape)
        # print(x_1.shape)

   

        # x_2 = inputs*torch.sigmoid(x_c)
        # print(x_2.shape)

        # x=x_1+x_2+inputs
        x=x_1+inputs

        # print(x.shape)
        x=self.linear(inputs)


        # torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        # torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        # x_max=x_a+x_max


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated 

class PFNLayer4_chanl(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=True):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            
            # self.avg=Aten(in_channels,out_channels)
            # self.sum=Aten(in_channels,out_channels)

            self.linear = nn.Linear(in_channels, out_channels, bias=False)

            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

        # self.point_att = PCAttention(32, reduction_rate=4, activation=nn.ReLU(),
                                        #  )
        
        self.channel_att = PCAttention(in_channels, reduction_rate=4, activation=nn.ReLU(),
                                           )
    def forward(self, inputs):

        # x_p = self.point_att(inputs.permute(0,2,1)).permute(0, 2, 1)
        # print(x_p.shape)
        x_c = self.channel_att(inputs)
        # print(x_c.shape)
        
        # x_1 = inputs*torch.sigmoid(x_p)
        # print(torch.sigmoid(x_p).shape)
        # print(x_1.shape)

   

        x_2 = inputs*torch.sigmoid(x_c)
        # print(x_2.shape)

        # x=x_1+x_2+inputs
        # x=x_1+inputs
        x=x_2+inputs


        # print(x.shape)
        x=self.linear(inputs)


        # torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        # torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # print(x_max.shape)

       
       
        # x_max=x_a+x_max


        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated 
        
if __name__ == '__main__':
    
    x=torch.randn(22986, 32, 10)
    # T=TModule()
    # y=T(x)
    # print(y.size())
    T2=PFNLayer4(10,64)
    y=T2(x)
    print(y.size())