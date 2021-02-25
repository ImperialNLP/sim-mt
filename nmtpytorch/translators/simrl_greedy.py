import re
import logging
import itertools

import torch
import numpy as np

from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions
from ..utils.misc import load_pt_file
from ..config import Options
from .. import models
from .greedy import GreedySearch

# modifs to the main code by Julia Ive
# julia.ive84@gmail.com


logger = logging.getLogger('nmtpytorch')


class SimultaneousRLGreedySearch(GreedySearch):
    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 max_len=100, **kwargs):
        if batch_size != 1:
            logger.info(f'Ignoring batch_size {batch_size} for simultaneous greedy search')
            batch_size = 1

        assert not model.opts.model['enc_bidirectional'], \
            "Bidirectional models can not be used for simultaneous MT."

        assert model.opts.model.get('dec_init', 'zero') == 'zero', \
            "`dec_init` should be 'zero' for simplicity."

        super().__init__(model, data_loader, out_prefix,
                         batch_size, filter_chain, max_len)

        data = load_pt_file(self.model.opts.model['env_file'])

        weights, _, opts = data['model'], data['history'], data['opts']
        opts = Options.from_dict(opts)
        instance = getattr(models, opts.train['model_type'])(opts=opts)

        # Setup layers
        instance.setup(is_train=False)

        # Load weights
        instance.load_state_dict(weights, strict=False)

        # Move to device
        instance.to(DEVICE)

        # Switch to eval mode
        instance.train(False)

        self.model_trans = instance

        # Partial modality i.e. text
        self._partial_key = str(model.sl)
        self.buffer = None

        self.list_of_s_0 = kwargs.pop('s_0', '').split(',')
        self.list_of_delta = kwargs.pop('delta', '').split(',')
        self.criteria = kwargs.pop('criteria', '').split(',')

    @staticmethod
    def wait_if_diff(cur_log_p, cur_next_pred, cand_log_p, cand_next_pred):
        """If the same word predicted with more context, WRITE. Otherwise READ."""
        return cur_next_pred.ne(cand_next_pred)

    @staticmethod
    def wait_if_worse(cur_log_p, cur_next_pred, cand_log_p, cand_next_pred):
        """If confidence for the current token increases, WRITE. Otherwise READ."""
        return cur_log_p[0, cur_next_pred] > cand_log_p[0, cur_next_pred]

    def write(self, new_word, new_h):
        """Write the new word, move the pointer and accept the hidden state."""
        self.prev_word, self.buffer[self.t_ptr] = new_word, new_word
        self.prev_h = new_h
        self.actions.append('1')
        self.t_ptr += 1
        self.eos_written = new_word.item() == self.eos

    def update_s(self, increment):
        """Update read pointer."""
        new_pos = min(self.s_len, self.s_ptr + increment)
        n_reads = new_pos - self.s_ptr
        self.actions.extend(['0'] * n_reads)
        self.s_ptr = new_pos

    def clear_states(self):
        self.s_ptr = 0
        self.t_ptr = 0
        self.prev_h = None
        self._c_states = None
        self.prev_word = None
        self.eos_written = False
        self.actions = []

        if self.buffer is None:
            # Write buffer
            self.buffer = torch.zeros(
                (self.max_len, ), dtype=torch.long, device=DEVICE)
        else:
            # Reset hypothesis buffer
            self.buffer.zero_()

    def cache_encoder_states(self, batch):
        """Encode source sentence full and cache it."""
        self.model.cache_enc_states(batch)
        self.s_len = batch[self._partial_key].size(0)

    def read_more(self, n):
        """Reads more source words and computes new states."""
        return self.model.get_enc_state_dict(up_to=self.s_ptr + n)

    def run_all(self):
        """Do a grid search over the given list of parameters."""
        #############
        # grid search
        #############
        settings = itertools.product(
            self.list_of_s_0,
            self.list_of_delta,
            self.criteria,
        )
        for s_0, delta, crit in settings:
            # Run the decoding
            hyps, actions, up_time = self.run()

            # Dumps two files one with segmentations preserved, another
            # with post-processing filters applied
            self.dump_results(hyps, suffix=f's{s_0}_d{delta}_{crit}')

            # Dump actions
            self.dump_lines(actions, suffix=f's{s_0}_d{delta}_{crit}.acts')

    def run(self, **kwargs):

        all_actions = []
        all_translations = []
        c = 0
        for batch in pbar(self.data_loader, unit='batch'):
            c += 1
            batch_length = batch.size

            #JI: max decoding length
            max_len = 40
            actions = np.full((batch_length, max_len), 3)
            #JI: arrays to store read counters and boolean to seq end
            src_read_counter = [1] * batch_length
            end_of_seq = [False] * batch_length

            src_max_len = batch[self._partial_key].size(0)
            vocab = self.model_trans.trg_vocab

            eos = vocab['<eos>']
            unk = vocab['<unk>']

            end_of_seq_ind = [39] * batch_length
            translations = np.full((batch_length, max_len), eos)
            batch.device(DEVICE)

            self.model_trans.cache_enc_states(batch)
            ctx_dict = self.model_trans.get_enc_state_dict()

            # JI: take the first state for the first action
            state_dict_full = ctx_dict[self._partial_key][0]
            state_dict_prep = state_dict_full[0].unsqueeze(0)

            state_dict = state_dict_prep.clone()
            state_dict_mask = torch.ones(1, batch_length).to(DEVICE)

            # JI: Compute decoder's initial state h_0 for each sentence (BxE)
            prev_h = self.model_trans.dec.f_init(ctx_dict)

            # Start all sentences with <s>
            prev_word_ind = self.model_trans.get_bos(batch_length).to(DEVICE)
            prev_word = self.model_trans.dec.emb(prev_word_ind)

            action = torch.zeros(batch_length, device=DEVICE)
            action_all = torch.zeros([batch_length, 2], device=DEVICE)

            if self.model.opts.model['mm_agent_init']:
                h_policy = self.model.imgctx2hid(
                    batch['image'].permute(1, 0, 2).reshape(batch_length, -1))
            else:
                h_policy = torch.zeros(
                    batch_length, self.model.opts.model['dec_dim'], device=DEVICE)

            if self.model.opts.model['mm_agent_att']:
                self.model.cache_enc_states(batch)
                # get image encoded state_dicts for RL agent
                image_ctx_vec = self.model.get_enc_state_dict()
            else:
                image_ctx_vec = None

            trg_step = 0

            # JI: loop until we generate <eos> or hit max
            while (not all(end_of_seq)) and trg_step < max_len:
                #JI: get next word
                if self.model.opts.model['mm_env']:
                    state_dict_in = {'src': (state_dict, state_dict_mask), 'image': ctx_dict['image']}
                else:
                    state_dict_in = {'src': (state_dict, state_dict_mask)}

                logp, new_h, ctx_vec = self.model_trans.dec.f_next(
                    state_dict_in, prev_word, prev_h)

                new_word_ind = logp.argmax(1)
                with torch.no_grad():
                    new_word = self.model_trans.dec.emb(new_word_ind)
                #JI: concat before input to actor
                policy_input = torch.cat((action_all.float(), new_word, ctx_vec), 1)

                #JI: get actor output
                out_dict = self.model.dec(h_policy, policy_input, image_ctx_vec, new_word, sample=False)

                h_policy = out_dict['h']
                action = out_dict['action']
                action_all = out_dict['action_all']

                state_dict_new = torch.zeros(
                    1, batch_length, self.model.opts.model['enc_dim']).to(DEVICE)
                state_dict_mask_new = torch.zeros(1, batch_length).to(DEVICE)

                #JI: we augment the dict size to accommodate new encoder states if our read counter tell us so
                if state_dict.size()[0] <= max(src_read_counter):
                    state_dict = torch.cat((state_dict, state_dict_new), 0)
                    state_dict_mask = torch.cat((state_dict_mask, state_dict_mask_new), 0)

                #JI: get clones for inplace operations
                prev_word_cl = prev_word.clone()
                if prev_h is None:
                    prev_h = new_h
                prev_h_cl = prev_h.clone()
                state_dict_cl = state_dict.clone()
                state_dict_mask_cl = state_dict_mask.clone()

                for n in range(batch_length):
                    action_sent = action[n].item()
                    # JI: if we took all the encoder states or action is WRITE

                    if src_read_counter[n] >= src_max_len or action_sent == 1:
                        # JI: we update prev word
                        prev_word_cl[n] = new_word[n]

                        # JI: we add our new word
                        translations[n, trg_step] = new_word_ind[n].item()

                        prev_h_cl[n] = new_h[n]

                        if end_of_seq[n] is False:
                            actions[n, trg_step] = 1

                        if new_word_ind[n] == eos:
                            end_of_seq[n] = True
                            end_of_seq_ind[n] = trg_step

                    else:
                        # JI: otherwise we read next word
                        state_dict_cl[src_read_counter[n]][n] = state_dict_full[src_read_counter[n]][n]
                        state_dict_mask_cl[-1][n] = 1
                        src_read_counter[n] += 1

                        # JI: temporary solution for now
                        # JI: for decoding, we do not add UNK to the hypothesis
                        translations[n, trg_step] = unk
                        if end_of_seq[n] is False:
                            actions[n, trg_step] = 0

                # JI: update initial values
                prev_h = prev_h_cl
                prev_word = prev_word_cl
                state_dict = state_dict_cl
                state_dict_mask = state_dict_mask_cl
                trg_step += 1

            for i, action in enumerate(actions):
                all_actions.append('0 ' + ' '.join(str(x) for x in action))

            for translation in translations:
                translation_txt = vocab.idxs_to_sent(translation)
                translation_txt = re.sub("<unk>", "", translation_txt)
                translation_txt = re.sub("\\s+", " ", translation_txt)
                all_translations.append(vocab.idxs_to_sent(translation))

        return (sort_predictions(self.data_loader, all_translations), sort_predictions(self.data_loader, all_actions), 0)
