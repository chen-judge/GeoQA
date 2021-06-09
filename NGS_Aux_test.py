from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU

from ManualProgram.eval_equ import Equations

import random
import warnings
import math
warnings.filterwarnings("ignore")

from utils import *

from mcan import *

import json


@Model.register("geo_s2s")
class SimpleSeq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    use_bleu : ``bool``, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 knowledge_points_ratio = 0,
                 attention: Attention = True,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 resnet_pretrained = None,
                 use_bleu: bool = True) -> None:
        super(SimpleSeq2Seq, self).__init__(vocab)

        resnet = build_model()

        if resnet_pretrained is not None:
            resnet.load_state_dict(torch.load(resnet_pretrained))
            print('##### Checkpoint Loaded! #####')
        else:
            print("No Diagram Pretrain !!!")
        self.resnet = resnet

        self.channel_transform = torch.nn.Linear(1024, 512)

        __C = Cfgs()
        self.mcan = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)  # not use

        self.decode_transform = torch.nn.Linear(1024, 512)

        self._equ = Equations()

        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(ngram_weights=(1, 0, 0, 0), exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None
        self._acc = Average()
        self._no_result = Average()

        # remember to clear after evaluation
        self.new_acc = []
        self.angle = []
        self.length = []
        self.area = []
        self.other = []
        self.point_acc_list = []
        self.save_results = dict()

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        # TODO: attention
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = LegacyAttention()
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None
            print("No Attention!")
            exit()

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

        # knowledge points
        self.point_ratio = knowledge_points_ratio
        if self.point_ratio != 0:
            self.points_norm = LayerNorm(__C.FLAT_OUT_SIZE)
            self.points_proj = nn.Linear(__C.FLAT_OUT_SIZE, 50)
            self.points_criterion = nn.BCELoss()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(self,  # type: ignore
                image, source_nums, choice_nums, label, type, data_id, manual_program,
                source_tokens: Dict[str, torch.LongTensor],
                point_label = None,
                target_tokens: Dict[str, torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        bs = len(label)
        state = self._encode(source_tokens)

        with torch.no_grad():
            img_feats = self.resnet(image)
        # (N, C, 14, 14) -> (N, 196, C)
        img_feats = img_feats.reshape(img_feats.shape[0], img_feats.shape[1], -1).transpose(1, 2)
        img_mask = make_mask(img_feats)
        img_feats = self.channel_transform(img_feats)

        lang_feats = state['encoder_outputs']
        # mask the digital encoding question without embedding, i.e. source_tokens(already index to number)
        lang_mask = make_mask(source_tokens['tokens'].unsqueeze(2))

        _, img_feats = self.mcan(lang_feats, img_feats, lang_mask, img_mask)

        # (N, 308, 512)
        # for attention, image first and then lang, using mask
        state['encoder_outputs'] = torch.cat([img_feats, lang_feats], 1)

        # decode
        state = self._init_decoder_state(state, lang_feats, img_feats, img_mask)
        output_dict = self._forward_loop(state, target_tokens)  # recurrent decoding for LSTM

        # knowledge points
        if self.point_ratio != 0:
            concat_feature = state["concat_feature"]
            point_feat = self.points_norm(concat_feature)
            point_feat = self.points_proj(point_feat)
            point_pred = torch.sigmoid(point_feat)
            point_loss = self.points_criterion(point_pred, point_label) * self.point_ratio
            output_dict["point_pred"] = point_pred
            output_dict["point_loss"] = point_loss
            output_dict["loss"] += point_loss


        # if testing, beam search and evaluation
        if not self.training:
            # state = self._init_decoder_state(state)
            state = self._init_decoder_state(state, lang_feats, img_feats, img_mask)  # TODO
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]

                # execute the decode programs to calculate the accuracy
                suc_knt, no_knt, = 0, 0

                selected_programs = []

                for b in range(bs):
                    hypo = None
                    used_hypo = None
                    choice = None
                    for i in range(self._beam_search.beam_size):
                        if choice is not None:
                            break
                        hypo = list(top_k_predictions[b][i])
                        if self._end_index in list(hypo):
                            hypo = hypo[:hypo.index(self._end_index)]
                        hypo = [self.vocab.get_token_from_index(idx.item()) for idx in hypo]
                        res = self._equ.excuate_equation(hypo, source_nums[b])
                        if res is not None and len(res) > 0:
                            for j in range(4):
                                if choice_nums[b][j] is not None and math.fabs(res[-1] - choice_nums[b][j]) < 0.001:
                                    choice = j
                                    used_hypo = hypo
                    selected_programs.append([hypo])

                    if choice is None:
                        no_knt += 1
                        answer_state = 'no_result'
                        self.new_acc.append(0)
                    elif choice == label[b]:
                        suc_knt += 1
                        answer_state = 'right'
                        self.new_acc.append(1)
                    else:
                        answer_state = 'false'
                        self.new_acc.append(0)

                    self.save_results[data_id[b]] = dict(manual_program=manual_program[b],
                                                         predict_program=hypo, predict_res=res,
                                                         choice=choice_nums[b], right_answer=label[b],
                                                         answer_state=answer_state)

                    flag = 1 if choice == label[b] else 0
                    if type[b] == 'angle':
                        self.angle.append(flag)
                    elif type[b] == 'length':
                        self.length.append(flag)
                    else:
                        self.other.append(flag)

                    # knowledge points
                    # if self.point_ratio != 0:
                    #     point_acc = self.multi_label_evaluation(point_pred[b].unsqueeze(0), point_label[b].unsqueeze(0))
                    #     self.point_acc_list.append(point_acc)

                # with open('save/test.json', 'w') as f:
                #    json.dump(self.save_results, f)

                if random.random() < 0.05:
                    print('selected_programs', selected_programs)

                # calculate BLEU
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])
                self._acc(suc_knt / bs)
                self._no_result(no_knt / bs)

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        img_mask = torch.ones(source_mask.shape[0], 196).long().cuda()
        concat_mask = torch.cat([img_mask, source_mask], 1)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,  # source_mask,
                "concat_mask": concat_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state, lang_feats, img_feats, img_mask):

        batch_size = state["source_mask"].size(0)
        final_lang_feat = util.get_final_encoder_states(
                        lang_feats,
                        state["source_mask"],
                        self._encoder.is_bidirectional())
        img_feat = self.attflat_img(img_feats, img_mask)
        feat = torch.cat([final_lang_feat, img_feat], 1)
        feat = self.decode_transform(feat)
        state["concat_feature"] = feat

        state["decoder_hidden"] = feat
        # C0 shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = torch.zeros(batch_size, self._decoder_output_dim).cuda()
        # state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)
        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            # recurrent decoding
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        # source_mask = state["source_mask"]
        source_mask = state["concat_mask"]

        # decoder_hidden and decoder_context are get from encoder_outputs in _init_decoder_state()
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)

        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    def multi_label_evaluation(self, input, target):
        one = torch.ones(target.shape).cuda()
        zero = torch.zeros(target.shape).cuda()
        res = torch.where(input > 0.5, one, zero)

        over = (res * target).sum(dim=1)
        union = res.sum(dim=1) + target.sum(dim=1) - over
        acc = over / union

        index = torch.isnan(acc)  # nan appear when both pred and target are zeros, which means makes right answer
        acc_fix = torch.where(index, torch.ones(acc.shape).cuda(), acc)

        acc_sum = acc_fix.sum().item()

        return acc_sum

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        # all_metrics.update({'acc': self._acc.get_metric(reset=reset)})
        all_metrics.update({'acc': self._acc.get_metric(reset=reset)})
        if len(self.new_acc) != 0:
            all_metrics.update({'new_acc': sum(self.new_acc)/len(self.new_acc)})
        print('Num of total, angle, len, other', len(self.new_acc), len(self.angle), len(self.length), len(self.other))
        if len(self.angle) != 0:
            all_metrics.update({'angle_acc': sum(self.angle)/len(self.angle)})
        if len(self.length) != 0:
            all_metrics.update({'length_acc': sum(self.length)/len(self.length)})
        if len(self.other) != 0:
            all_metrics.update({'other_acc': sum(self.other)/len(self.other)})
        all_metrics.update({'no_result': self._no_result.get_metric(reset=reset)})

        # if len(self.point_acc_list) != 0:
        #     all_metrics.update({'point_acc': sum(self.point_acc_list) / len(self.point_acc_list)})

        return all_metrics
