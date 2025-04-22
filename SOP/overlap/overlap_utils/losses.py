import torch
import torch.nn as nn
from overlap.overlap_utils.preprocessGPU import compute_overlaps


class ModelLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ModelLoss, self).__init__()
        self.eps = eps
        self.overlap_loss = nn.CrossEntropyLoss(reduction='mean')
        # self.overlap_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def matching_loss(self, scores, matching_mask):
        '''
        Calculating loss for coarse node matching
        :param scores: Predicted matching matrix
        :param matching_mask: The ground truth soft matching mask
        :return: Calculated loss
        '''
        mask_scores = matching_mask * scores
        loss = torch.sum(-mask_scores) / torch.sum(matching_mask)
        return loss

    def calc_recall(self, scores, matching_mask, thres=0.1):
        '''
        Calculate Recall for predicted correspondences
        :param scores: Predicted matching matrix
        :param matching_mask: Ground truth soft matching matrix
        :param thres: Threshold for sampling correspondences from matching matrix
        :return: Calculated precision
        '''
        mask = (matching_mask > 0)
        scores = torch.exp(scores[:-1, :-1])
        score_mask = (scores > thres).float() * mask[:-1, :-1]
        return score_mask.sum() / matching_mask.sum()

    def overlap_losses(self, input_dict, pre_x_ol, pre_y_ol):

        p = len(input_dict['stack_lengths']) - 1
        len_node = input_dict['stack_lengths'][-1][0]
        input_dict['overlap_pyr'] = compute_overlaps(input_dict)
        node_overlap = torch.reshape(input_dict['overlap_pyr'][f'pyr_{p}'], (1, -1))
        src_overlap_p = (node_overlap[:, :len_node] != 0).type(torch.long)
        tgt_overlap_p = (node_overlap[:, len_node:] != 0).type(torch.long)
        src_overlap_loss = self.overlap_loss(pre_x_ol, src_overlap_p)
        tgt_overlap_loss = self.overlap_loss(pre_y_ol, tgt_overlap_p)
        overlap_loss = (src_overlap_loss + tgt_overlap_loss) / 2

        return overlap_loss

    def local_matching_loss(self, scores, matching_mask):
        '''
        Calculating loss for fine point matching
        :param scores: Predicted matching matrix
        :param matching_mask: The ground truth soft matching mask
        :return: Calculated loss
        '''
        mask_scores = matching_mask * scores
        loss = torch.sum(-mask_scores) / torch.sum(matching_mask)
        return loss

    def calc_precision(self, scores, matching_mask, thres=0.5):
        '''
        Calculate Precision for predicted correspondences
        :param scores: Predicted matching matrix
        :param matching_mask: Ground truth soft matching matrix
        :param thres: Threshold for sampling correspondences from matching matrix
        :return: Calculated precision
        '''
        argmax_row = torch.argmax(scores[:-1, :], dim=1)
        argmax_col = torch.argmax(scores[:, :-1], dim=0)

        row_mask = torch.zeros_like(matching_mask)
        col_mask = torch.zeros_like(matching_mask)

        row_idx = torch.arange(end=scores.shape[0] - 1)
        col_idx = torch.arange(end=scores.shape[1] - 1)


        row_mask[row_idx, argmax_row] = 1.
        col_mask[argmax_col, col_idx] = 1.

        mask = (matching_mask > 0)

        prediction = (row_mask + col_mask) > thres

        correct = torch.sum(prediction[:-1, :-1] * mask[:-1, :-1])

        total = torch.sum(prediction[:-1, :-1]) + 1e-8

        return correct / total

    def forward(self, input_dict, scores, matching_mask, pre_x_ol, pre_y_ol, local_scores, local_scores_gt):
        stats = {}
        matching_loss = self.matching_loss(scores, matching_mask)
        overlap_loss = self.overlap_losses(input_dict, pre_x_ol, pre_y_ol)
        local_matching_loss = self.local_matching_loss(local_scores, local_scores_gt)
        total_loss = matching_loss + overlap_loss + local_matching_loss
        matching_recall = self.calc_recall(scores.detach().cpu(), matching_mask.cpu())
        # 点匹配精度
        local_matching_precision = 0.
        for i in range(local_scores.shape[0]):
            local_matching_precision += self.calc_precision(local_scores[i].detach().cpu(),
                                                            local_scores_gt[i].detach().cpu())
        local_matching_precision /= local_scores.shape[0]
        stats['matching_loss'] = matching_loss
        stats['matching_recall'] = matching_recall
        stats['overlap_loss'] = overlap_loss
        stats['local_matching_loss'] = local_matching_loss
        stats['local_matching_precision'] = local_matching_precision
        stats['total_loss'] = total_loss
        return stats
