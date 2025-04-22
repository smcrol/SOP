import numpy as np
import torch
import torch.nn as nn
from model.KPConv.models import KPEncoder, KPDecoder
from model.AttentionModules.cross import CrossAttention, PointEncoder
from model.OptimalTransport.utils import log_optimal_transport, rpmnet_sinkhorn
from lib.utils import correspondences_from_thres, knn, square_distance

from overlap.CG import CGModule

import cpp_wrappers.grouping.lib.grouping_cuda as grouping


class RoughMatchingModel(nn.Module):
    def __init__(self, config):
        super(RoughMatchingModel, self).__init__()
        self.encoder = KPEncoder(config=config)
        self.decoder = KPDecoder(config=config)

        gnn_feats_dim = config.gnn_feats_dim  # gnn_feats_dim: 256
        local_gnn_feats_dim = config.intermediate_feats_dim  # intermediate_feats_dim: 32

        self.ape = config.ape  # whether to apply positional embeddings
        self.acn = config.acn  # wheter to apply conditional network
        if self.ape:
            self.position = PointEncoder(gnn_feats_dim, [32, 64, 128, 256])
        ##############################################
        # attention part, self-cross-self attetion
        self.sattn1 = CrossAttention(feature_dim=gnn_feats_dim, num_heads=config.num_head)
        self.cattn = CrossAttention(feature_dim=gnn_feats_dim, num_heads=config.num_head)
        self.sattn2 = CrossAttention(feature_dim=gnn_feats_dim, num_heads=config.num_head)
        ##############################################
        self.final_proj = nn.Conv1d(gnn_feats_dim, gnn_feats_dim, kernel_size=1, bias=True)
        self.sinkhorn_iters = config.sinkhorn_iters
        self.node_id = config.node_id
        self.test = (config.mode == 'test')
        self.corr_sel = config.corr_sel
        self.neighbor_sel = config.neighbor_sel
        self.initial_thres = config.initial_thres
        self.thres_decay = config.thres_decay
        self.min_coarse_corr = config.min_coarse_corr
        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        l_bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('l_bin_score', l_bin_score)

        #################################################
        # CG Model part
        self.CG = CGModule(True)
        ##############
        # local conditional network, local attention part
        if self.acn:
            self.l_sattn1 = CrossAttention(feature_dim=local_gnn_feats_dim, num_heads=config.num_head)
            self.l_cattn = CrossAttention(feature_dim=local_gnn_feats_dim, num_heads=config.num_head)
            self.l_sattn2 = CrossAttention(feature_dim=local_gnn_feats_dim, num_heads=config.num_head)
        #################################################
        self.l_final_proj = nn.Conv1d(local_gnn_feats_dim, local_gnn_feats_dim, kernel_size=1, bias=True)
        self.pos_margin = config.pos_margin

    def forward(self, batch):
        len_src_c = batch['stack_lengths'][0][0]  # (17629,)
        rot = batch['rot']  #
        trans = batch['trans']  #
        pcd_c = batch['points'][0]  # (31168,3)
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]  # (17629,3);(13539,3)
        src_pcd_raw, tgt_pcd_raw = batch['src_pcd_raw'], batch['tgt_pcd_raw']  # (17629,3);(13539,3)

        src_pcd_raw_tsfm = (torch.matmul(rot, src_pcd_raw.T) + trans).T  # (17629,3);

        len_node_c = batch['stack_lengths'][-1][0]  # (313,)
        node_c = batch['points'][-1]  # (529,3)
        src_node_c, tgt_node_c = node_c[:len_node_c], node_c[len_node_c:]  # (313,3);(216,3)
        #####################################
        # Region split and grouping
        src_pcd_id, tgt_pcd_id = (knn(src_pcd_c.unsqueeze(0), src_node_c.unsqueeze(0), 1),
                                  knn(tgt_pcd_c.unsqueeze(0), tgt_node_c.unsqueeze(0), 1))  # (1,17629,1);(1,13539,1)
        src_patch = -torch.ones((1, src_node_c.shape[0], self.neighbor_sel)).cuda().int()  # (1,313,64)
        grouping.grouping_wrapper(int(src_patch.shape[0]), int(src_pcd_id.shape[1]), int(src_patch.shape[1]), int(src_patch.shape[2]),
                                  src_pcd_id.int(), src_patch.int())
        # [B, N, k]
        tgt_patch = -torch.ones((1, tgt_node_c.shape[0], self.neighbor_sel)).cuda().int()  # (1,216,64)
        # [B, M, k]
        grouping.grouping_wrapper(int(tgt_patch.shape[0]), int(tgt_pcd_id.shape[1]), int(tgt_patch.shape[1]), int(tgt_patch.shape[2]),
                                  tgt_pcd_id.int(), tgt_patch.int())

        src_val_mask = torch.zeros_like(src_patch, dtype=torch.long).cuda()  # (1,313,64)
        src_val_mask[src_patch < 0] = 1
        src_patch[src_patch < 0] = 0

        tgt_val_mask = torch.zeros_like(tgt_patch, dtype=torch.long).cuda()  # (1,216,64)
        tgt_val_mask[tgt_patch < 0] = 1
        tgt_patch[tgt_patch < 0] = 0

        ######################################
        # 1. Joint encoder part
        feats_c, skip_x = self.encoder(batch)  # (1,256,529);
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_node_c], feats_c[:, :, len_node_c:]  # (1,256,313);(1,256,216)

        #######################################
        # 2. Self-cross-self attention part
        if self.ape:
            centered_src_node_c = src_node_c - torch.mean(src_node_c, dim=-1, keepdim=True)
            centered_tgt_node_c = tgt_node_c - torch.mean(tgt_node_c, dim=-1, keepdim=True)
            src_feats_c = src_feats_c + self.position(centered_src_node_c)
            tgt_feats_c = tgt_feats_c + self.position(centered_tgt_node_c)

        # self
        src_feats_c = src_feats_c + self.sattn1(src_feats_c, src_feats_c)
        tgt_feats_c = tgt_feats_c + self.sattn1(tgt_feats_c, tgt_feats_c)
        # cross
        src_feats_c = src_feats_c + self.cattn(src_feats_c, tgt_feats_c)
        tgt_feats_c = tgt_feats_c + self.cattn(tgt_feats_c, src_feats_c)

        # self
        src_feats_c = src_feats_c + self.sattn2(src_feats_c, src_feats_c)
        tgt_feats_c = tgt_feats_c + self.sattn2(tgt_feats_c, tgt_feats_c)

        # final proj
        src_feats_c = self.final_proj(src_feats_c)  # (1,256,313)
        tgt_feats_c = self.final_proj(tgt_feats_c)  # (1,256,216)
        #######################################
        # 3. CG overlap_mask part and Sinkhorn Optimal Transport part
        acg = True
        if acg:
            x_ol, y_ol, src_mask_feature, tgt_mask_feature = self.CG(src_feats_c, tgt_feats_c)
        else:
            src_mask_feature = src_feats_c
            tgt_mask_feature = tgt_feats_c
            x_ol, y_ol = None, None

        dim = src_mask_feature.size(1)  # (256,)
        scores = torch.einsum('bdn,bdm->bnm', src_mask_feature, tgt_mask_feature)  # (1,313,216)
        scores = scores / dim ** 0.5  # 归一化得分

        scores = log_optimal_transport(scores, None, None, self.bin_score, iters=self.sinkhorn_iters)  # (1,314,217)(每个数值表示两个分布之间元素配对的对数概率)

        #########################################
        # 4. Decoder part
        decoder_input_feats = torch.cat([src_feats_c, tgt_feats_c], dim=-1)
        final_feats = self.decoder(batch, decoder_input_feats, skip_x)  # (31168,32)
        src_final_f, tgt_final_f = final_feats[:len_src_c, :], final_feats[len_src_c:, :]  # (17692,32);(13539,32)

        ##########################################
        # 5. Local Sinkhorn Optimal Transport part
        if not self.test:
            node_corr = batch['node_corr']  # (529,2)
        else:
            begin_thres = self.initial_thres
            while True:
                node_corr, node_corr_conf = correspondences_from_thres(score=scores[0], thres=begin_thres, supp=True,
                                                                       return_score=True)
                if node_corr.shape[0] >= self.min_coarse_corr:
                    break
                begin_thres -= self.thres_decay

        if not self.test:
            # Sampling a fixed number of coarse correspondences for training
            index = np.arange(node_corr.shape[0])
            index_sel = torch.from_numpy(np.random.choice(index, size=self.corr_sel)).long().cuda()
            node_corr = node_corr[index_sel]  # (128,2)

        src_node_sel, tgt_node_sel = node_corr[:, 0].long(), node_corr[:, 1].long()  # (128,) (128,)
        src_pcd_sel = src_patch[0, src_node_sel, :].view(-1).long()  # (8192); 128x64
        tgt_pcd_sel = tgt_patch[0, tgt_node_sel, :].view(-1).long()
        src_node_mask = src_val_mask[0, src_node_sel, :]  # (128,64)
        tgt_node_mask = tgt_val_mask[0, tgt_node_sel, :]

        if not self.test:
            tgt_pcd_raw_tsfm = tgt_pcd_raw.clone()
            src_pcd_sel_c, tgt_pcd_sel_c = src_pcd_raw_tsfm[src_pcd_sel, :], tgt_pcd_raw_tsfm[tgt_pcd_sel, :]
            src_pcd_sel_c, tgt_pcd_sel_c = (src_pcd_sel_c.view(node_corr.shape[0], self.neighbor_sel, -1),
                                            tgt_pcd_sel_c.view(node_corr.shape[0], self.neighbor_sel, -1))  # (128,64,3)

            distance = torch.sqrt(square_distance(src_pcd_sel_c, tgt_pcd_sel_c))  # (128,64,64)
            local_scores_gt = (distance < self.pos_margin).float()  # pos_margin = 0.1

            local_scores_row = torch.clamp(1. - torch.sum(local_scores_gt, dim=-1), min=0.).unsqueeze(-1)  # (128,64,1)
            local_scores_col = torch.clamp(1. - torch.sum(local_scores_gt, dim=-2), min=0.).unsqueeze(-2)  # (128,1,64)
            supp = torch.zeros(size=(local_scores_gt.shape[0], 1, 1), dtype=torch.float32).cuda()  # (128,1,1)
            local_scores_col = torch.cat([local_scores_col, supp], dim=-1)  # (128,1,65)
            local_scores_gt = torch.cat([local_scores_gt, local_scores_row], dim=-1)  # (128,64,65)
            local_scores_gt = torch.cat([local_scores_gt, local_scores_col], dim=-2)  # (128,65,65)
            # 利用掩码忽略矩阵中不参与匹配的值
            local_scores_gt[:, :-1, :] *= (1 - src_node_mask.unsqueeze(2).expand(src_node_mask.shape[0], src_node_mask.shape[1], src_node_mask.shape[1] + 1))
            local_scores_gt[:, :, :-1] *= (1 - tgt_node_mask.unsqueeze(1).expand(tgt_node_mask.shape[0], tgt_node_mask.shape[1] + 1, tgt_node_mask.shape[1]))
            src_pcd_sel_f, tgt_pcd_sel_f = src_final_f[src_pcd_sel, :], tgt_final_f[tgt_pcd_sel, :]  # (8192,32)
            src_pcd_sel_f, tgt_pcd_sel_f = (src_pcd_sel_f.view(node_corr.shape[0], self.neighbor_sel, -1).transpose(2, 1),
                                            tgt_pcd_sel_f.view(node_corr.shape[0], self.neighbor_sel, -1).transpose(2, 1))  # (128,32,64) (N, F, 64) 其中 N 是节点对应关系的数量，F 是特征的数量;为了适应torch.einsum操作。

        else:
            num_corr = node_corr.shape[0]
            local_scores_gt = None
            src_pcd_c_tsfm = src_pcd_raw.clone()
            tgt_pcd_c_tsfm = tgt_pcd_raw.clone()
            src_pcd_sel_c, tgt_pcd_sel_c = src_pcd_c_tsfm[src_pcd_sel, :], tgt_pcd_c_tsfm[tgt_pcd_sel, :]
            src_pcd_sel_c, tgt_pcd_sel_c = src_pcd_sel_c.view(num_corr, self.neighbor_sel, -1), tgt_pcd_sel_c.view(num_corr, self.neighbor_sel, -1)
            src_pcd_sel_f, tgt_pcd_sel_f = src_final_f[src_pcd_sel, :], tgt_final_f[tgt_pcd_sel, :]
            src_pcd_sel_f, tgt_pcd_sel_f = (src_pcd_sel_f.view(num_corr, self.neighbor_sel, -1).transpose(2, 1),
                                            tgt_pcd_sel_f.view(num_corr, self.neighbor_sel, -1).transpose(2, 1))

        if self.acn:
            src_pcd_sel_f = src_pcd_sel_f + self.l_sattn1(src_pcd_sel_f, src_pcd_sel_f)
            tgt_pcd_sel_f = tgt_pcd_sel_f + self.l_sattn1(tgt_pcd_sel_f, tgt_pcd_sel_f)
            src_pcd_sel_f = src_pcd_sel_f + self.l_cattn(src_pcd_sel_f, tgt_pcd_sel_f)
            tgt_pcd_sel_f = tgt_pcd_sel_f + self.l_cattn(tgt_pcd_sel_f, src_pcd_sel_f)
            src_pcd_sel_f = src_pcd_sel_f + self.l_sattn2(src_pcd_sel_f, src_pcd_sel_f)
            tgt_pcd_sel_f = tgt_pcd_sel_f + self.l_sattn2(tgt_pcd_sel_f, tgt_pcd_sel_f)

        src_pcd_sel_f = self.l_final_proj(src_pcd_sel_f)  # (128,32,64)
        tgt_pcd_sel_f = self.l_final_proj(tgt_pcd_sel_f)
        local_dim = final_feats.size(1)
        local_scores = torch.einsum('bdn,bdm->bnm', src_pcd_sel_f, tgt_pcd_sel_f)  # (128,64,64)每个批次中的128组，都有一个64x64的矩阵
        local_scores = local_scores / local_dim ** 0.5  # (128,64,64)
        row_score_mask = src_node_mask.unsqueeze(2).expand(src_node_mask.shape[0], src_node_mask.shape[1],
                                                           src_node_mask.shape[1])  # (128,64,64)
        col_score_mask = tgt_node_mask.unsqueeze(1).expand(tgt_node_mask.shape[0], tgt_node_mask.shape[1],
                                                           tgt_node_mask.shape[1])  # (128,64,64)
        bins0 = torch.tensor([0]).cuda().repeat(src_node_mask.shape[0], src_node_mask.shape[1])  # (128,64)
        bins0 = bins0.unsqueeze(2)  # (128,64,1)

        bins1 = torch.tensor([0]).cuda().repeat(tgt_node_mask.shape[0], tgt_node_mask.shape[1])  # (128,64)
        bins1 = bins1.unsqueeze(1)  # (128,1,64)
        local_scores[row_score_mask > 0] = -1e6  # 将similarity matrix中的invalid entries（即点数不足64的情况）用-∞进行屏蔽
        local_scores[col_score_mask > 0] = -1e6

        local_scores = rpmnet_sinkhorn(local_scores, bins0, bins1, iters=self.sinkhorn_iters)  # (128,65,65)

        if not self.test:
            return scores, x_ol, y_ol, local_scores, local_scores_gt
        else:
            return src_pcd_sel_c.view(-1, 3), tgt_pcd_sel_c.view(-1, 3), local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel, \
                   x_ol, y_ol, src_patch, tgt_patch
