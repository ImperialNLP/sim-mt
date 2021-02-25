import torch.nn.functional as F
from torch import nn, zeros


class SupervisedAttentionLoss(nn.Module):

    def __init__(self, loss_func='nll', ignore_index=-1, loss_weight=1.0):
        super(SupervisedAttentionLoss, self).__init__()
        self._loss_func = loss_func
        self.ignore_index = ignore_index
        self._loss_weight = loss_weight

    def forward(self, predictions, alignments):
        """
        Computes the loss.
        :param predictions: The predictions of shape (batch_size, seq_len, num_objects).
                            We have a maximum of num_objects, which in the case of the entities is 30, indexed from 0.
                            If should_apply_log is True, a log will be applied to obtain log probabilities.
        :param alignments: The alignments values of shape (batch_size, seq_len, max_aligned_objs).
                           Contain the indices of the objects within which a word is aligned,
                           where -1 is used for padding the tensors and does not correspond to a valid object.
                           Example: batch_size = 1, sequence_length = 4, max_aligned_objs = 2, num_objects = 5
                           [[
                           [0, 3],
                           [2, -1],
                           [1, -1],
                           [-1, -1]
                           ]].
                           The first word is aligned to the first and forth object
                           The second word is aligned only to the third object,
                           The third word is aligned to only to the second object.
                           The forth word has no corresponding objects to which it's aligned.
                           This means that the target probability attention distribution is:
                           [[
                           [0.5, 0., 0., 0.5, 0.],
                           [0., 0., 1.0, 0., 0.],
                           [0., 1.0, 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           ]].
        :return: The computed loss.
        """
        # Get the locations of the valid object indices. This will gives a 3 element tuple.
        # The 0th tuple element contains the indices of dim 0 (batch_size),
        # the 1st tuple element the indices of dim 1 (seq_len)
        # and the 2nd tuple element the indices at dim 2 (max_aligned_objs)
        valid_obj_locations = (alignments != self.ignore_index).nonzero(as_tuple=True)
        obj_indices = alignments[valid_obj_locations]

        model_prob = self._create_model_prob(valid_obj_locations, obj_indices, predictions)

        predictions = predictions.clamp(min=1e-18).log()
        if self._loss_func == 'kl_div':
            total_loss = F.kl_div(predictions, model_prob, reduction='sum')
        else:
            total_loss = -(predictions.mul(model_prob)).sum()

        total_loss = total_loss.mul(self._loss_weight)
        return total_loss.div(obj_indices.size().numel())

    @staticmethod
    def _create_model_prob(valid_obj_locations, obj_indices, predictions):
        # In order to match the way the attention probabilities are shaped which is (batch_size, seq_len, num_objects)
        # we need to flag the correct object regions for a given word in a batch
        model_prob = zeros(predictions.shape).to(predictions.device)
        prob_indices = (valid_obj_locations[0], valid_obj_locations[1], obj_indices)
        model_prob[prob_indices] = 1.0

        # Normalize to get soft labels
        row_sum = model_prob.sum(-1, keepdim=True)
        # Threshold to avoid divisions by zero as there are words that don't have any aligned objects
        F.threshold(row_sum, 1, 1, inplace=True)
        model_prob.div_(row_sum)
        return model_prob
