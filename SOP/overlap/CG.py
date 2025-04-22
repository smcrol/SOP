import torch
import torch.nn as nn
import torch.nn.functional as F



class PointNet(nn.Module):
    def __init__(self, in_dim, gn, out_dims, cls=False):
        super(PointNet, self).__init__()
        self.cls = cls
        l = len(out_dims)
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                         nn.GroupNorm(8, out_dim))
            if self.cls and i != l - 1:
                self.backbone.add_module(f'pointnet_relu_{i}',
                                         nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone(x)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)
        return f, g


class MLPs(nn.Module):
    def __init__(self, in_dim, mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()
        l = len(mlps)
        for i, out_dim in enumerate(mlps):
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if i != l - 1:
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)
        return x


class CGModule(nn.Module):
    def __init__(self, gn):
        super(CGModule, self).__init__()
        self.decoder_ol = PointNet(in_dim=256,
                                   gn=gn,
                                   out_dims=[512, 512, 256, 2],
                                   cls=True)

    def forward(self, src_feature, tgt_feature):
        '''
        Context-Guided Model for initial alignment and overlap score.
        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :return: T0: (B, 3, 4), OX: (B, N, 2), OY: (B, M, 2)
        '''

        # overlap prediction and overlapping mask prediction
        x_ol = self.decoder_ol(src_feature, pooling=False)
        y_ol = self.decoder_ol(tgt_feature, pooling=False)

        x_ol_mask = torch.argmax(x_ol, dim=1)
        y_ol_mask = torch.argmax(y_ol, dim=1)

        x_ol_mask = x_ol_mask.unsqueeze(1)
        y_ol_mask = y_ol_mask.unsqueeze(1)

        # overlap_mask
        src_mask_feature = src_feature * x_ol_mask
        tgt_mask_feature = tgt_feature * y_ol_mask

        return x_ol, y_ol, src_mask_feature, tgt_mask_feature
