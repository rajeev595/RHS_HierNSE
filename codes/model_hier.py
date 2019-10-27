# -*- coding: utf-8 -*-
__author__ = "Rajeev Bhatt Ambati"

from HierNSE import HierNSE
from utils import average_gradients, get_running_avg_loss
import params as params
from beam_search import run_beam_search_hier
from rouge_batch import rouge_l_fscore_batch as rouge_l_fscore

import random
import math
import time

import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMStateTuple
import numpy as np
from joblib import Parallel, delayed

tf.set_random_seed(2019)
random.seed(2019)
tf.reset_default_graph()

FLAGS = tf.app.flags.FLAGS


class SummarizationModelHier(object):
    def __init__(self, vocab, data):
        self._vocab = vocab
        self._data = data

        self._dense_init = tf.contrib.layers.xavier_initializer()

        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True

        self._best_val_loss = np.infty
        self._num_gpus = len(FLAGS.GPUs)

        self._sess = None
        self._saver = None
        self._init = None

    def _create_placeholders(self):
        """
            This function creates the placeholders needed for the computation graph.
            [enc_in, enc_pad_mask, enc_doc_mask, dec_in, enc_in_ext_vocab, labels, dec_pad_mask]
        :return:
        """
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Word embedding.
        if FLAGS.use_pretrained:
            with tf.variable_scope("embed", reuse=tf.AUTO_REUSE):
                self._word_embedding = tf.get_variable(name="word_embedding",
                                                       shape=[self._vocab.size(), FLAGS.dim],
                                                       initializer=tf.constant_initializer(self._vocab.wvecs),
                                                       dtype=tf.float32,
                                                       trainable=True)

        else:
            self._word_embedding = tf.get_variable(name="word_embedding",
                                                   shape=[self._vocab.size(), FLAGS.dim],
                                                   dtype=tf.float32,
                                                   trainable=True)

        # Graph Inputs/Outputs.
        self._enc_in = tf.placeholder(dtype=tf.int32, name='enc_in',
                                      shape=[FLAGS.batch_size, FLAGS.max_enc_sent,
                                             FLAGS.max_enc_steps_per_sent])                 # Shape: B x S_in x T_in.
        self._enc_pad_mask = tf.placeholder(dtype=tf.float32, name='enc_pad_mask',
                                            shape=[FLAGS.batch_size, FLAGS.max_enc_sent,
                                                   FLAGS.max_enc_steps_per_sent])           # Shape: B x S_in x T_in.
        self._enc_doc_mask = tf.placeholder(dtype=tf.float32, name='enc_doc_mask',
                                            shape=[FLAGS.batch_size, FLAGS.max_enc_sent])   # Shape: B x S_in.
        self._dec_in = tf.placeholder(dtype=tf.int32, name='dec_in',
                                      shape=[FLAGS.batch_size, FLAGS.dec_steps])            # Shape: B x T_dec.

        inputs = [self._enc_in, self._enc_pad_mask, self._enc_doc_mask, self._dec_in]

        # Additional inputs in pointer-generator mode.
        if FLAGS.use_pgen:
            self._enc_in_ext_vocab = tf.placeholder(dtype=tf.int32,
                                                    name='enc_inp_ext_vocab',
                                                    shape=[FLAGS.batch_size,
                                                           FLAGS.max_enc_sent *
                                                           FLAGS.max_enc_steps_per_sent])   # Shape: B x T_enc.
            self._max_oov_size = tf.placeholder(dtype=tf.int32, name='max_oov_size')
            inputs.append(self._enc_in_ext_vocab)

        # Additional ground-truth's when in training.
        if FLAGS.mode.lower() == "train":
            self._dec_out = tf.placeholder(dtype=tf.int32, name='dec_out',
                                           shape=[FLAGS.batch_size, FLAGS.dec_steps])           # Shape: B x T_dec.
            self._dec_pad_mask = tf.placeholder(dtype=tf.float32, name='dec_pad_mask',
                                                shape=[FLAGS.batch_size, FLAGS.dec_steps])      # Shape: B x T_dec.
            inputs += [self._dec_out, self._dec_pad_mask]

        return inputs

    def _create_writers(self):
        """
            This function creates the summaries and writers needed for visualization through tensorboard.
        :return: writers.
        """
        self._mean_crossentropy_loss = tf.placeholder(dtype=tf.float32, name='mean_crossentropy_loss')
        self._mean_rouge_score = tf.placeholder(dtype=tf.float32, name="mean_rouge")

        # Summaries.
        cross_entropy_summary = tf.summary.scalar('cross_entropy', self._mean_crossentropy_loss)
        rouge_summary = None
        if FLAGS.rouge_summary:
            rouge_summary = tf.summary.scalar('rouge_score', self._mean_rouge_score)
        self._summaries = tf.summary.merge_all()

        summary_list = [cross_entropy_summary]
        if FLAGS.rouge_summary:
            summary_list.append(rouge_summary)

        self._val_summaries = tf.summary.merge(summary_list, name="validation_summaries")

        # Summary writers.
        self._train_writer = tf.summary.FileWriter(FLAGS.PathToTB + 'train')    # , self._sess.graph)
        self._val_writer = tf.summary.FileWriter(FLAGS.PathToTB + 'val')

    def build_graph(self):
        start = time.time()
        if FLAGS.mode == 'train':
            if len(FLAGS.GPUs) > 1:
                self._parallel_model()  # Parallel model in case of multiple-GPUs.
            else:
                self._single_model()    # Single model for a single GPU/CPU.

        if FLAGS.mode == 'test':
            inputs = self._create_placeholders()                    # [enc_in, dec_in, enc_in_ext_vocab]
            self._forward(inputs)                                   # Predictions Shape: Bm x 1

            self._saver = tf.train.Saver(tf.global_variables())     # Saver.
            self._init = tf.global_variables_initializer()          # Initializer.
            self._sess = tf.Session(config=self._config)            # Session.

            # Restoring the trained model.
            self._saver.restore(self._sess, FLAGS.PathToCheckpoint)

        end = time.time()
        print("build_graph took %.2f sec. \n" % (end - start))

    def _forward(self, inputs):
        """
            This function creates the TensorFlow computation graph.
        :param inputs: A list of input placeholders.
                [0] enc_in: Encoder input sequence of ID's, Shape: B x S_in x T_in.
                [1] enc_pad_mask: Encoder input mask to indicate the presence of PAd tokens, Shape: B x S_in x T_in.
                [2] enc_doc_mask: Encoder document mask to indicate the presence of empty sentences, Shape: B x S_in.
                [3] dec_in: Decoder input sequence of ID's, Shape: B x T_dec.

                    Following additional input in pointer-generator mode.
                [4] enc_in_ext_vocab: Encoder input representation in the extended vocabulary, Shape: B x T_enc.

                [5] labels: Ground-Truth labels, Only in train mode, Shape: B x T_dec.
                [6] dec_pad_mask: Decoder output mask, Shape: B x T_dec.
        :return: returns loss in train mode and predictions in test mode.
        """
        batch_size = inputs[0].get_shape().as_list()[0]     # Batch-size
        # The NSE instance
        self._nse = HierNSE(batch_size=batch_size, dim=FLAGS.dim, dense_init=self._dense_init,
                            mode=FLAGS.mode, use_comp_lstm=FLAGS.use_comp_lstm, num_layers=FLAGS.num_layers)

        # Encoder, used while testing phase.
        self._prev_states = self._encoder(inputs[: 3])

        # Decoder.
        outputs = self._decoder(inputs[1: 5] + [inputs[-1]], self._prev_states)

        if FLAGS.mode.lower() == "test":
            self._topk_ids, self._topk_log_probs, self._curr_states, self._p_attns = outputs[: 4]

            if FLAGS.use_pgen:
                self._p_gens = outputs[4]

        else:
            probs, p_attns = outputs

            crossentropy_loss = self._get_crossentropy_loss(probs=probs, labels=inputs[-2], mask=inputs[-1])

            # Evaluating a few samples.
            samples = []
            if FLAGS.rouge_summary:
                samples = self._get_samples(inputs[1: 5], self._prev_states)

            return crossentropy_loss, samples

    def _encoder(self, inputs):
        """
            This is the encoder.
        :param inputs: A list of the following inputs.
                enc_in: Encoder input sequence of ID's, Shape: B x S_in x T_in.
                enc_pad_mask: Encoder input mask to indicate the presence of PAD tokens, Shape: B x S_in x T_in.
                enc_doc_mask: Encoder document mask to indicate the presence of empty sentences, Shape: B x S_in.
        :return:
        A list of internal states of NSE after the last encoding step.
            [0] memory: [sent_mems, doc_mem] The sentence and document memories respectively,
                        Shape: [B x S_in x T_in x D, B x S_in x D].
            [1] read_state: Hidden state of the read LSTM after the last encoding step, (c, h) Shape: 2 * [B x D].
            [2] write_state: Hidden state of the write LSTM after the last encoding step, (c, h) Shape: 2 * [B x D].
            [3] comp_state: Hidden state of the compose LSTM after the last encoding step, (c, h) Shape: 2 * [B x D].
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc_in, enc_pad_mask, enc_doc_mask = inputs

            # Converting ID's to word-vectors.
            enc_in_vecs = tf.nn.embedding_lookup(params=self._word_embedding, ids=enc_in)  # Shape: B x S_in x T_in x D.
            enc_in_vecs = tf.cast(enc_in_vecs, dtype=tf.float32)                           # Cast to float32.

            # Document memory.
            doc_mem = tf.reduce_mean(enc_in_vecs, axis=2, name="document_memory")   # Shape: B x S_in x D.

            new_sent_mems = []                                  # New sentence memories.
            lstm_states = [None, None, None]                    # LSTM states.

            for i in range(FLAGS.max_enc_sent):
                # Mask, memory of ith sentence.
                sent_i_mask = enc_pad_mask[:, i, :]             # Shape: B x T_in.
                sent_i_mem = enc_in_vecs[:, i, :, :]            # Shape: B x T_in x D.

                mem_masks = [sent_i_mask, enc_doc_mask]         # ith-sentence and document masks.
                state = [[sent_i_mem, doc_mem]] + lstm_states   # NSE internal state.

                for j in range(FLAGS.max_enc_steps_per_sent):
                    # j-th token from the ith sentence.
                    x_t = enc_in_vecs[:, i, j, :]   # Shape: B x D.
                    output, state = self._nse.step(x_t=x_t, mem_masks=mem_masks, prev_state=state)

                # Update
                new_sent_mems.append(state[0][0])           # ith-sentence memory.
                doc_mem = state[0][1]                       # Document memory.
                lstm_states = state[1:]                     # Read, write, compose states.

            new_sent_mems = tf.concat(new_sent_mems, axis=1)        # Shape: B x (S_in*T_in) x D.
            all_states = [[new_sent_mems, doc_mem]] + lstm_states

            return all_states

    def _decoder(self, inputs, all_states):
        """
            This is the decoder.
        :param inputs: A list of the following inputs.
                [0] enc_pad_mask: Encoder input mask to indicate the presence of PAD tokens, Shape: B x S_in x T_in.
                [1] enc_doc_mask: Encoder document mask to indicate the presence of empty sentences, Shape: B x S_in.
                [2] dec_in: Input to the decoder, Shape: B x T_dec.
                Following additional inputs in pointer generator mode:
                    [3] enc_in_ext_vocab: (For pointer generator mode)
                            Encoder input representation in the extended vocabulary, Shape: B x T_enc.
                [4] dec_pad_mask: Decoder mask to indicate the presence of PAD tokens, Shape: B x T_dec.

        :param all_states: The internal states of NSE after the last encoding step.
                [0] memory: [sent_mems, doc_mem] The sentence and document memories respectively,
                            Shape: [B x T_enc x D, B x S_in x D].
                [1] read_state: Hidden state of the read LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
                [2] write_state: Hidden state of the write LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
                [3] comp_state: Hidden state of the compose LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
        :return:
            In test mode:
                [topk_ids, topk_log_probs, state, p_attns] + [p_gens]
            In train mode:
                [p_cums, p_attns]
        """
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            enc_pad_mask, enc_doc_mask, dec_in = inputs[: 3]

            # Concatenating the pad masks of all sentences.
            enc_pad_mask = tf.reshape(
                enc_pad_mask, [-1, FLAGS.max_enc_sent*FLAGS.max_enc_steps_per_sent])        # Shape: B x T_enc.

            # Converting ID's to word vectors.
            dec_in_vecs = tf.nn.embedding_lookup(params=self._word_embedding, ids=dec_in)   # Shape: B x T_dec x D.
            dec_in_vecs = tf.cast(dec_in_vecs, dtype=tf.float32)                            # Shape: B x T_dec x D.

            # Memory masks.
            mem_masks = [enc_pad_mask, enc_doc_mask]            # Shape: [B x T_enc, B x S_in].

            # NSE internal states.
            sent_mems, doc_mem = all_states[0]                  # Shape: B x T_enc x D, B x S_in x D.
            state = [[sent_mems, doc_mem]] + all_states[1:]

            writes = []
            p_attns = []
            p_gens = []
            for i in range(FLAGS.dec_steps):

                x_t = dec_in_vecs[:, i, :]                                                      # Shape: B x D.
                output, state = self._nse.step(
                    x_t=x_t, mem_masks=mem_masks, prev_state=state, use_pgen=FLAGS.use_pgen
                )

                # Appending the outputs.
                writes.append(output[0])
                p_attns.append(output[1])

                if FLAGS.use_pgen:
                    p_gens.append(output[2])

            p_vocabs = self._get_vocab_dist(writes)                             # Shape: T_dec * [B x vsize].

            p_cums = p_vocabs
            if FLAGS.use_pgen:
                enc_in_ext_vocab = inputs[3]
                p_cums = self._get_cumulative_dist(
                    p_vocabs, p_gens, p_attns, enc_in_ext_vocab
                )                                                               # Shape: T_dec * [B x ext_vsize].

            if FLAGS.mode.lower() == "test":
                p_final = p_cums[0]

                # The top k predictions, topk_ids, Shape: Bm * (2*Bm).
                # Respective probabilities, topk_probs, Shape: Bm * (2*Bm).
                topk_probs, topk_ids = tf.nn.top_k(p_final, k=2*FLAGS.beam_size, name="topk_preds")
                topk_log_probs = tf.log(tf.clip_by_value(topk_probs, 1e-10, 1.0))

                outputs = [topk_ids, topk_log_probs, state, p_attns]
                if FLAGS.use_pgen:
                    outputs.append(p_gens)

                return outputs
            else:
                return p_cums, p_attns

    def _get_samples(self, inputs, state):
        """
            This function samples greedily from decoder output to calculate the ROUGE score.
        :param inputs: A list of the following inputs.
                [0] enc_pad_mask: Encoder input mask to indicate the presence of PAD tokens, Shape: B x S_in x T_in.
                [1] enc_doc_mask: Encoder document mask to indicate the presence of empty sentences, Shape: B x S_in.
                [2] dec_in: Input to the decoder, Shape: B x T_dec.
                Following additional inputs in pointer generator mode:
                    [3] enc_in_ext_vocab: (For pointer generator mode)
                            Encoder input representation in the extended vocabulary, Shape: B x T_enc.
        :param state: The internal states of NSE after the last encoding step.
                [0] memory: [sent_mems, doc_mem] The sentence and document memories respectively,
                            Shape: [B x T_enc x D, B x S_in x D].
                [1] read_state: Hidden state of the read LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
                [2] write_state: Hidden state of the write LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
                [3] comp_state: Hidden state of the compose LSTM after the last encoding step,(c, h) Shape: 2 * [B x D].
        :return:
                probs: Probabilities used for sampling, Shape: B x T_dec x V.
                samples: The samples, Shape: B x T_dec.
        """
        unk_id = self._vocab.word2id(params.UNKNOWN_TOKEN)      # Unknown ID.
        start_id = self._vocab.word2id(params.START_DECODING)   # Start ID.

        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            enc_pad_mask, enc_doc_mask = inputs[: 2]
            batch_size = enc_pad_mask.get_shape().as_list()[0]
            # Concatenating the pad masks of all sentences.
            enc_pad_mask = tf.reshape(
                enc_pad_mask,
                [-1, FLAGS.max_enc_sent * FLAGS.max_enc_steps_per_sent])    # Shape: B x T_enc.

            # Memory masks.
            mem_masks = [enc_pad_mask, enc_doc_mask]                        # Shape: [B x T_enc, B x S_in].

            samples = []                                                    # Sampling outputs.
            for i in range(FLAGS.dec_steps):

                if i == 0:
                    id_t = tf.fill([batch_size], start_id)                  # Shape: B x .
                else:
                    id_t = samples[-1][:, 0]
                    # Replacing the ID's from external vocabulary (if any) to UNK id's.
                    id_t = tf.where(
                        tf.less(id_t, self._vocab.size()), id_t, unk_id * tf.ones_like(id_t)
                    )

                # Getting the word vector.
                x_t = tf.nn.embedding_lookup(params=self._word_embedding, ids=id_t)  # Shape: B x D.
                x_t = tf.cast(x_t, dtype=tf.float32)

                output, state = self._nse.step(
                    x_t=x_t, mem_masks=mem_masks, prev_state=state, use_pgen=FLAGS.use_pgen,
                )

                # Output probability distribution.
                p_vocab = self._get_vocab_dist([output[0]])                         # Shape: [B x vsize].

                # Calculate cumulative probability distribution using pointer mechanism.
                p_cum = p_vocab
                if FLAGS.use_pgen:
                    p_cum = self._get_cumulative_dist(
                        p_vocabs=p_vocab, p_gens=[output[2]], p_attns=[output[1]], enc_in_ext_vocab=inputs[-1]
                    )  # Shape: T_dec * [B x ext_vsize].

                # Greedy sampling.
                _, gs_sample = tf.nn.top_k(p_cum[0], k=FLAGS.num_samples)   # Shape: B x 1.
                samples.append(gs_sample)

            samples = tf.concat(samples, axis=1)                            # Shape: B x T_dec.

            return samples

    def run_encoder(self, inputs):
        """
            This function calculates the internal states of NSE after the last step of encoding.
        :param inputs:
                    enc_in_batch: A batch of encoder input sequence, Shape: Bm x S_in x T_in.
                    enc_pad_mask: Encoder input mask, Shape: Bm x S_in x T_in.
                    enc_doc_mask: Encoder document mask, Shape: Bm x S_in.
        :return:
            NSE internal states after encoding:
                final_memory: [sent_mems, doc_mem] The sentence and document memories respectively after last time-step
                              of encoding. Shape: [1 x T_enc x D, 1 x S_in x D].
                final_read_state: Hidden state of read LSTM after last time-step of encoding,(c, h)
                                  Shape: (1 x D, 1 x D).
                final_write_state: Hidden state of write LSTM after last time-step of encoding,(c, h)
                                   Shape: (1 x D, 1 x D).
                final_comp_state: Hidden state of compose LSTM after last time-step of encoding,(c, h),
                                  Shape: (1 x 3D, 1 x 3D).
            Attention distributions: sentence, document attention.
                T_enc * [1 x T_enc, 1 x S_in].

        """
        to_return = [self._prev_states[0][0], self._prev_states[0][1]] + self._prev_states[1:]

        outputs = self._sess.run(to_return, feed_dict={self._enc_in: inputs[0],
                                                       self._enc_pad_mask: inputs[1],
                                                       self._enc_doc_mask: inputs[2]})
        final_memory_s, final_memory_d = outputs[: 2]

        # memory Shape: [Bm x T_enc x D, Bm x S_in x D].
        # read state Shape: (Bm x D, Bm x D).
        # write state Shape: (Bm x D, Bm x D).
        final_read_state, final_write_state = outputs[2: 4]

        # Since the states repeated values, slicing only first one.
        final_memory_s = final_memory_s[0, np.newaxis, :, :]    # Shape: 1 x T_enc x D.
        final_memory_d = final_memory_d[0, np.newaxis, :, :]    # Shape: 1 x S_in x D.

        final_memory = [final_memory_s, final_memory_d]
        final_comp_state = None

        def get_state_slice(inp_states):
            if FLAGS.num_layers == 1:
                return LSTMStateTuple(inp_states.c[0, np.newaxis, :],
                                      inp_states.h[0, np.newaxis, :])

            else:
                sliced_states = []
                for _, layer_state in enumerate(inp_states):
                    if FLAGS.rnn.lower() == "lstm":
                        sliced_states.append(
                            LSTMStateTuple(layer_state.c[0, np.newaxis, :],
                                           layer_state.h[0, np.newaxis, :])
                        )
                    else:
                        sliced_states.append(
                            layer_state[0, np.newaxis, :]
                        )

                return tuple(sliced_states)

        final_read_state = get_state_slice(final_read_state)    # Shape: (1 x D, 1 x D).
        final_write_state = get_state_slice(final_write_state)  # Shape: (1 x D, 1 x D).
        if FLAGS.use_comp_lstm:
            final_comp_state = get_state_slice(outputs[4])      # Shape: (1 x 3D, 1 x 3D).

        state = [final_memory, final_read_state, final_write_state]
        if FLAGS.use_comp_lstm:
            state.append(final_comp_state)

        return state

    def decode_one_step(self, inputs, prev_states):
        """
            This function performs one step of decoding.
        :param inputs:
                [0] dec_in_batch:
                        The input to the decoder. This is the output from previous time-step, Shape: Bm * [1 x]
                [1] enc_pad_mask: Encoder input mask to indicate the presence of PAD tokens.
                                  Useful to calculate the attention over memory, Shape: Bm x T_enc.
                [2] enc_doc_mask: Encoder document mask to indicate the presence of empty sentences, Shape: Bm x S_in.
            In pointer generator mode, there are following additional inputs:
                [3]: enc_in_ex_vocab_batch:
                        Encoder input sequence represented in extended vocabulary, Shape: Bm x T_enc.
                [4]: max_oov_size: Size of the largest OOV tokens in the current batch, Shape: ()
        :param prev_states: previous internal states of NSE of all Bm hypothesis, Bm * [prev_state].
                            where state is a list of internal states of NSE for a single hypothesis.
                prev_state = [prev_memory, prev_read_state, prev_write_state]
                prev_memory: [sent_mems, doc_mem] The sentence and document memories respectively after last
                             previous time-step, Shape: [1 x T_enc x D, 1 x S_in x D].
                prev_read_state: Hidden state of read LSTM after previous time step, (c, h) Shape: [1 x D, 1 x D].
                prev_write_state: Hidden state of write LSTM after previous time step, (c, h) Shape: [1 x D, 1 x D].
                prev_comp_state: Hidden state of compose LSTM after previous time step, (c, h) Shape: [1 x 3D, 1 x 3D].
        :return:
            topk_ids: Top-k predictions in the current step, Shape: Bm x (2*Bm)
            topk_log_probs: log probabilities of top-k predictions, Shape: Bm x (2*Bm)
            curr_states: Current internal states of NSE, Bm * [state].
                [0]: memory: NSE sentence and document memories. [Bm x T_enc x D, Bm x S_in x D].
                [1]: read_state, (c, h) Shape: (Bm x D, Bm x D).
                [2]: write_state, (c, h) Shape: (Bm x D, Bm x D).
                [3]: comp_state, (c, h) Shape: (Bm x 3D, Bm x 3D).
            p_gens: Generation probabilities, Shape: Bm x .
            p_attns: Attention probabilities, Shape: [Bm x T_in, Bm x S_in].
        """
        # Decoder input
        dec_in = inputs[0]                                      # Shape: Bm * [1 x]
        dec_in = np.stack(dec_in, axis=0)                       # Shape: Bm x
        inputs[0] = np.expand_dims(dec_in, axis=-1)             # Shape: Bm x 1

        # Previous memories of Bm hypothesis.
        # Sentence memories.
        prev_memories_s = [state[0][0] for state in prev_states]    # Shape: Bm * [1 x T_enc x D].
        prev_memories_s = np.concatenate(prev_memories_s, axis=0)   # Shape: Bm x T_enc x D.

        # Document memory.
        prev_memories_d = [state[0][1] for state in prev_states]    # Shape: Bm * [1 x S_in x D].
        prev_memories_d = np.concatenate(prev_memories_d, axis=0)   # Shape: Bm x S_in x D.

        prev_memories = [prev_memories_s, prev_memories_d]

        def get_combined_states(inp_states):
            """
                A function to combine the states of Bm hypothesis.
            :param inp_states: List of states of Bm hypothesis. Bm * [(s1, s2, ..., s_l)]
            :return:
            """
            if FLAGS.num_layers == 1:
                # Cell states.
                combined_states_c = [hyp_state.c for hyp_state in inp_states]   # Shape: Bm * [1 x D].
                combined_states_c = np.concatenate(combined_states_c, axis=0)   # Shape: Bm x D.

                # Hidden states.
                combined_states_h = [hyp_state.h for hyp_state in inp_states]   # Shape: Bm * [1 x D].
                combined_states_h = np.concatenate(combined_states_h, axis=0)   # Shape: Bm x D.

                combined_states = LSTMStateTuple(combined_states_c,
                                                 combined_states_h)             # Shape: (Bm x D, Bm x D).

            else:
                combined_states = []
                for i in range(FLAGS.num_layers):
                    # Cell states.
                    layer_state_c = [layer_state[i].c for layer_state in inp_states]    # Shape: Bm * [1 x D].
                    layer_state_c = np.concatenate(layer_state_c, axis=0)               # Shape: Bm x D.

                    # Hidden states.
                    layer_state_h = [layer_state[i].h for layer_state in inp_states]    # Shape: Bm * [1 x D].
                    layer_state_h = np.concatenate(layer_state_h, axis=0)               # Shape: Bm x D.

                    combined_states.append(
                        LSTMStateTuple(layer_state_c, layer_state_h)
                    )

                combined_states = tuple(combined_states)

            return combined_states

        prev_read_states = get_combined_states([state[1] for state in prev_states])
        prev_write_states = get_combined_states([state[2] for state in prev_states])
        if FLAGS.use_comp_lstm:
            prev_comp_states = get_combined_states([state[3] for state in prev_states])
        else:
            prev_comp_states = None

        feed_dict = {
            self._dec_in: inputs[0],
            self._enc_pad_mask: inputs[1],
            self._enc_doc_mask: inputs[2],
            self._prev_states[0][0]: prev_memories[0],
            self._prev_states[0][1]: prev_memories[1],
            self._prev_states[1]: prev_read_states,
            self._prev_states[2]: prev_write_states
        }

        if FLAGS.use_comp_lstm:
            feed_dict[self._prev_states[3]] = prev_comp_states

        if FLAGS.use_pgen:
            feed_dict[self._enc_in_ext_vocab] = inputs[3]
            feed_dict[self._max_oov_size] = inputs[4]

        to_return = [self._topk_ids, self._topk_log_probs, self._curr_states[0][0], self._curr_states[0][1],
                     self._curr_states[1], self._curr_states[2]]

        if FLAGS.use_comp_lstm:
            to_return.append(self._curr_states[3])

        if FLAGS.use_pgen:
            to_return += [self._p_gens, self._p_attns]

        outputs = self._sess.run(to_return, feed_dict=feed_dict)

        # Preparing the next values (inputs and states for next time step).
        next_values = outputs[: 2]

        # Current memories.
        curr_memories_s = np.split(outputs[2], FLAGS.beam_size, axis=0)   # Shape: Bm * [1 x T_enc x D].
        curr_memories_d = np.split(outputs[3], FLAGS.beam_size, axis=0)   # Shape: Bm * [1 x S_in x D].
        curr_memories = [[mem_s, mem_d] for mem_s, mem_d in
                         zip(curr_memories_s, curr_memories_d)]    # Shape: Bm * [[1 x T_enc x D, 1 x S_in x D]].

        def get_states_split(inp_states):
            """
                This function splits the states for Bm hypothesis.
            :param inp_states:
                    if just one layer:
                        A NumPy array of shape Bm x D.
                    for multiple layers:
                        A tuple of states of RNN layers (s1, s2, ...., s_l).
            :return:
            """
            if FLAGS.num_layers == 1:
                split_states_c = np.split(inp_states.c, FLAGS.beam_size, axis=0)  # Shape: Bm * [1 x D].
                split_states_h = np.split(inp_states.h, FLAGS.beam_size, axis=0)  # Shape: Bm * [1 x D].
                split_states = [LSTMStateTuple(c, h)
                                for c, h in zip(split_states_c, split_states_h)]    # Shape: Bm * [(1 x D, 1 x D)].

            else:
                split_states = []
                for i in range(FLAGS.beam_size):
                    hyp_state_c = [layer_state.c[i, np.newaxis, :]
                                   for layer_state in inp_states]               # num_layers * [1 x D].
                    hyp_state_h = [layer_state.h[i, np.newaxis, :]
                                   for layer_state in inp_states]               # num_layers * [1 x D].
                    hyp_state = [LSTMStateTuple(c, h)
                                 for c, h in zip(hyp_state_c, hyp_state_h)]     # num_layers * [(1 x D, 1 x D)]
                    split_states.append(tuple(hyp_state))

            return split_states

        # Split the states for Bm hypothesis.
        curr_read_states = get_states_split(outputs[4])
        curr_write_states = get_states_split(outputs[5])
        if FLAGS.use_comp_lstm:
            curr_comp_states = get_states_split(outputs[6])
        else:
            curr_comp_states = None

        if FLAGS.use_comp_lstm:

            curr_states_list = [[memory, read_state, write_state, comp_state]
                                for memory, read_state, write_state, comp_state in
                                zip(curr_memories, curr_read_states, curr_write_states, curr_comp_states)]
        else:
            # Forming a list of internal states for Bm hypothesis.
            curr_states_list = [[memory, read_state, write_state] for memory, read_state, write_state
                                in zip(curr_memories, curr_read_states, curr_write_states)]

        next_values.append(curr_states_list)

        # Generation probabilities.
        if FLAGS.use_pgen:
            p_gens = outputs[-2]                                    # Shape: 1 x [Bm x 1].
            p_gens = p_gens[0]                                    # Only one time-step in decoding phase, Shape: Bm x 1.
            p_gens = np.squeeze(p_gens, axis=1)                     # Shape: Bm x
            p_gens = np.split(p_gens, FLAGS.beam_size, axis=0)      # Shape: Bm * [1]
            next_values.append(p_gens)

        # Attention probabilities.
        if FLAGS.use_pgen:
            p_attns = outputs[-1]                                       # Shape: 1 x [[Bm x T_enc, Bm x S_in]].
            p_attns = p_attns[0]             # Only one time-step in the decoding phase, Shape: [Bm x T_enc, Bm x S_in].
            p_attns_s = np.split(p_attns[0], FLAGS.beam_size, axis=0)   # Shape: Bm * [1 x T_enc].
            p_attns_d = np.split(p_attns[1], FLAGS.beam_size, axis=0)   # Shape: Bm * [1 x S_in].
            p_attns = [[attn_s, attn_d] for attn_s, attn_d
                       in zip(p_attns_s, p_attns_d)]                    # Shape: Bm * [[1 x T_enc, 1 x S_in]].
            next_values.append(p_attns)

        return next_values

    def _get_vocab_dist(self, inputs):
        """
            This function passes the NSE hidden states obtained from decoder through the output layer (dense layer
            followed by a softmax layer) and returns the vocabulary distribution thus obtained.
        :param inputs: List of hidden states, Shape: T_dec * [B x D]
        :return: p_vocab: List of vocabulary distributions for each time-step, Shape: T_dec * [B x vsize].
        """
        steps = len(inputs)
        vsize = self._vocab.size()

        inputs = tf.concat(inputs, axis=0)                                              # Shape: (B*T_dec) x D.

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            scores = tf.layers.dense(inputs,
                                     units=vsize,
                                     activation=None,
                                     kernel_initializer=self._dense_init,
                                     kernel_regularizer=None,
                                     name='output')                                     # Shape: (B*T_dec) x vsize.

            p_vocab = tf.nn.softmax(scores, name="prob_scores")                         # Shape: (B*T_dec) x vsize.

        p_vocab = tf.split(p_vocab, num_or_size_splits=steps, axis=0)                   # Shape: T_dec * [B x vsize].

        return p_vocab

    def _get_cumulative_dist(self, p_vocabs, p_gens, p_attns, enc_in_ext_vocab):
        """
            This function calculates the cumulative probability distribution from the vocabulary distribution, attention
            distribution and generation probabilities.
        :param p_vocabs: A list of vocabulary distributions for each time-step, Shape: T_dec * [B x vsize].
        :param p_gens: A list of generation probabilities for each time-step, Shape: T_dec * [B x 1]
        :param p_attns: A list of attention distributions for each time-step, T_dec  * [[sent_attn, doc_attn]],
                        Shape: T_dec * [[B x T_enc, B x S_in]].
        :param enc_in_ext_vocab: Encoder input represented using extended vocabulary, Shape: B x T_enc.
        :return:
        """
        vsize = self._vocab.size()
        ext_vsize = vsize + self._max_oov_size
        batch_size, enc_steps = enc_in_ext_vocab.get_shape().as_list()

        p_vocabs = [tf.multiply(p_gen, p_vocab)
                    for p_gen, p_vocab in zip(p_gens, p_vocabs)]    # Shape: T_dec * [B x vsize].
        p_attns = [tf.multiply(tf.subtract(1.0, p_gen), p_attn[0])
                   for p_gen, p_attn in zip(p_gens, p_attns)]       # Shape: T_dec * [B x T_enc].

        zero_vocab = tf.zeros(shape=[batch_size, self._max_oov_size])
        # Shape: T_dec * [B x ext_vsize].
        p_vocabs_ext = [tf.concat([p_vocab, zero_vocab], axis=-1) for p_vocab in p_vocabs]

        idx = tf.range(0, limit=batch_size)                     # Shape: B x.
        idx = tf.expand_dims(idx, axis=-1)                      # Shape: B x 1.
        idx = tf.tile(idx, [1, enc_steps])                      # Shape: B x T_enc.
        indices = tf.stack([idx, enc_in_ext_vocab], axis=-1)     # Shape: B x T_enc x 2.

        # First, A zero matrix of shape B x ext_vsize is created. Then, the attention score for each input token
        # indexed as in indices is looked in p_attn and updated in the corresponding location,
        # Shape: T_dec * [B x ext_vsize]
        p_attn_cum = [tf.scatter_nd(indices, p_attn, [batch_size, ext_vsize]) for p_attn in p_attns]

        # Cumulative distribution, Shape: T_dec * [B x ext_vsize].
        p_cum = [tf.add(gen_prob, copy_prob) for gen_prob, copy_prob in zip(p_vocabs_ext, p_attn_cum)]

        return p_cum

    @ staticmethod
    def _get_crossentropy_loss(probs, labels, mask):
        """
            This function calculates the crossentropy loss from ground-truth and the probability scores.
        :param probs: Predicted probabilities, Shape: T * [B x vocab_size]
        :param labels: Ground Truth labels, Shape: B x T.
        :param mask: Mask to exclude PAD tokens while calculating loss, Shape: B x T.
        :return: average mini-batch loss.
        """
        if type(probs) is list:
            probs = tf.stack(probs, axis=1)                                     # Shape: B x T x vsize.

        true_probs = tf.reduce_sum(
            probs * tf.one_hot(labels, depth=tf.shape(probs)[2]), axis=2
        )                                                                       # Shape: B x T.
        logprobs = -tf.log(tf.clip_by_value(true_probs, 1e-10, 1.0))            # Shape: B x T.
        xe_loss = tf.reduce_sum(logprobs * mask) / tf.reduce_sum(mask)

        return xe_loss

    def _restore(self):
        """
            This function restores parameters from a saved checkpoint.
        :return:
        """
        restore_path = FLAGS.PathToCheckpoint + "model_epoch10"

        if FLAGS.restore_checkpoint and tf.train.checkpoint_exists(restore_path):
            start = time.time()

            # Initializing all variables.
            print("Initializing all variables!\n")
            self._sess.run(self._init)

            # Restoring checkpoint variables.
            vars_restore = [v for v in self._global_vars if "Adam" not in v.name]
            restore_saver = tf.train.Saver(vars_restore)
            print("Restoring non-Adam parameters from a previous checkpoint.\n")
            restore_saver.restore(self._sess, restore_path)

            end = time.time()
            print("Restoring model took %.2f sec. \n" % (end - start))
        else:
            start = time.time()
            self._init.run(session=self._sess)
            end = time.time()
            print("Running initializer took %.2f time. \n" % (end - start))

    def train(self):
        """
            This function performs the training followed by validation after every few epochs and saves the best model
            if validation loss is decreased. It also writes the training summaries for visualization using tensorboard.
        :return:
        """
        # Restore from previous checkpoint if found one.
        self._restore()

        # No. of iterations.
        num_train_iters_per_epoch = int(math.ceil(self._data.num_train_examples / FLAGS.batch_size))

        # Running averages.
        running_avg_crossentropy_loss = 0.0
        running_avg_rouge = 0.0
        for epoch in range(1, 1 + FLAGS.num_epochs):
            # Training.
            for iteration in range(1, 1 + num_train_iters_per_epoch):
                start = time.time()
                feed_dict = self._get_feed_dict(split="train")

                to_return = [self._train_op, self._crossentropy_loss, self._global_step, self._summaries]
                if FLAGS.rouge_summary:
                    to_return.append(self._samples)

                # Evaluate summaries for last batch.
                feed_dict[self._mean_crossentropy_loss] = running_avg_crossentropy_loss
                if FLAGS.rouge_summary:
                    feed_dict[self._mean_rouge_score] = running_avg_rouge

                outputs = self._sess.run(to_return, feed_dict=feed_dict)
                _, crossentropy_loss, global_step = outputs[: 3]

                # Calculating ROUGE score.
                rouge_score = 0.0
                if FLAGS.rouge_summary:
                    samples = outputs[4]
                    lens = np.sum(feed_dict[self._dec_pad_mask], axis=1).astype(np.int32)  # Shape: B x .
                    gt_labels = feed_dict[self._dec_out]
                    rouge_score = rouge_l_fscore(samples, gt_labels, None, lens, False)

                # Updating the running averages.
                running_avg_crossentropy_loss = get_running_avg_loss(
                    crossentropy_loss, running_avg_crossentropy_loss
                )
                running_avg_rouge = get_running_avg_loss(
                    np.mean(rouge_score), running_avg_rouge
                )

                if ((iteration - 2) % FLAGS.summary_every == 0) or (iteration == num_train_iters_per_epoch):
                    train_summary = outputs[3]
                    self._train_writer.add_summary(train_summary, global_step)

                end = time.time()
                print("\rTraining Iteration: {}/{} ({:.1f}%) took {:.2f} sec.".format(
                    iteration, num_train_iters_per_epoch, iteration * 100 / num_train_iters_per_epoch, end - start
                ))

            if epoch % FLAGS.val_every == 0:
                start = time.time()
                self._validate()
                end = time.time()
                print("Validation took {:.2f} sec.".format(end - start))

    def _validate(self):
        """
            This function validates the saved model.
        :return:
        """
        # Validation
        num_val_iters_per_epoch = int(math.ceil(self._data.num_val_examples / FLAGS.val_batch_size))
        num_train_iters_per_epoch = int(math.ceil(self._data.num_train_examples / FLAGS.batch_size))

        outputs = None
        total_crossentropy_loss = 0.0
        total_rouge_score = 0.0
        for iteration in range(1, 1 + num_val_iters_per_epoch):
            feed_dict = self._get_feed_dict(split="val")
            to_return = [self._crossentropy_loss]
            if FLAGS.rouge_summary:
                to_return.append(self._samples)

            if iteration == num_val_iters_per_epoch:
                feed_dict[self._mean_crossentropy_loss] = total_crossentropy_loss / (num_val_iters_per_epoch - 1)
                feed_dict[self._mean_rouge_score] = total_rouge_score / (num_val_iters_per_epoch - 1)
                to_return += [self._val_summaries, self._global_step]

            outputs = self._sess.run(to_return, feed_dict=feed_dict)
            crossentropy_loss = outputs[0]

            # Calculating ROUGE score.
            rouge_score = 0.0
            if FLAGS.rouge_summary:
                samples = outputs[1]
                lens = np.sum(feed_dict[self._dec_pad_mask], axis=1).astype(np.int32)  # Shape: B x .
                gt_labels = feed_dict[self._dec_out]
                rouge_score = rouge_l_fscore(samples, gt_labels, None, lens, False)

            print("\rValidation Iteration: {}/{} ({:.1f}%)".format(
                iteration, num_val_iters_per_epoch, iteration * 100 / num_val_iters_per_epoch,
            ))

            # Updating the total losses.
            total_crossentropy_loss += crossentropy_loss
            total_rouge_score += np.mean(rouge_score)

        # Writing the validation summaries for visualization.
        val_summary, global_step = outputs[-2], outputs[-1]
        self._val_writer.add_summary(val_summary, global_step)
        epoch = global_step // num_train_iters_per_epoch

        # Cumulative loss.
        avg_xe_loss = total_crossentropy_loss / num_val_iters_per_epoch
        if avg_xe_loss < self._best_val_loss:
            print("\rValidation loss improved :).")
        self._saver.save(self._sess, FLAGS.PathToCheckpoint + "model_epoch" + str(epoch))

    def test(self):
        """
            This function predicts the outputs for the test set.
        :return:
        """
        num_gpus = len(FLAGS.GPUs)                  # Total no. of GPUs.
        batch_size = num_gpus * FLAGS.num_pools     # Total no. of examples processed per a single test iteration.

        # No. of iterations
        num_test_iters_per_epoch = int(math.ceil(self._data.num_test_examples / batch_size))

        for iteration in range(1, 1 + num_test_iters_per_epoch):
            start = time.time()
            # [indices, summaries, files[start: end], enc_inp, enc_pad_mask, enc_doc_mask,
            # enc_inp_ext_vocab, ext_vocabs, max_oov_size]
            input_batches = self._data.get_batch(
                batch_size, split="test", permutate=FLAGS.permutate, chunk=FLAGS.chunk
            )

            # Split the inputs into batches of size "num_pools" per GPU.
            input_batches_per_gpu = [[] for _ in range(num_gpus)]
            for i, input_batch in enumerate(input_batches):
                for j, idx in zip(range(num_gpus), range(0, batch_size, FLAGS.num_pools)):

                    if i < 5 or (FLAGS.use_pgen and i < 7):             # First 5 inputs (7 inputs in pgen mode).
                        input_batches_per_gpu[j].append(
                            input_batch[idx: idx + FLAGS.num_pools])
                    else:                                               # max_oov_size is the same for all examples.
                        input_batches_per_gpu[j].append(input_batch)

            # Appending
            # Appending the GPU id.
            for gpu_id in range(num_gpus):
                input_batches_per_gpu[gpu_id].append(gpu_id)

            Parallel(n_jobs=num_gpus, backend="threading")(
                map(delayed(self._test_one_gpu), input_batches_per_gpu)
            )
            end = time.time()
            print("\rTesting Iteration: {}/{} ({:.1f}%) in {:.1f} sec.".format(
                iteration, num_test_iters_per_epoch, iteration * 100 / num_test_iters_per_epoch, end - start))

    def _test_one_gpu(self, inputs):
        """
            This function performs testing on the inputs of a single GPU.
        :return:
        """
        input_batches = inputs[: -1]
        gpu_idx = inputs[-1]

        with tf.device('/gpu:%d' % gpu_idx):
            self._test_one_pool(input_batches)

    def _test_one_pool(self, inputs):
        """
            This function performs testing on the inputs of a single GPU over parallel pools.
        :param inputs: All the first 8 inputs have "num_pools" examples each.
                    [indices, summaries, files[start: end], enc_inp, enc_pad_mask, enc_doc_mask,
                     enc_inp_ext_vocab, ext_vocabs, max_oov_size]
        :return:
        """
        # Split the inputs into each example per pool.
        input_batches_per_pool = [[] for _ in range(FLAGS.num_pools)]
        for i, input_batch in enumerate(inputs):
            for j in range(FLAGS.num_pools):
                # Splitting the GT summaries.
                if i == 0 or i == 1 or i == 2 or (FLAGS.use_pgen and i == 7):
                    input_batches_per_pool[j].append([input_batch[j]])

                # Repeating the input Numpy arrays for beam size no. of times.
                elif i == 3 or i == 4 or i == 5 or (FLAGS.use_pgen and i == 6):
                    input_batches_per_pool[j].append(
                        np.repeat(input_batch[np.newaxis, j], repeats=FLAGS.beam_size, axis=0))

                # max_oov_size is same for all examples in the pool.
                else:
                    input_batches_per_pool[j].append(input_batch)

        Parallel(n_jobs=FLAGS.num_pools, backend="threading")(
            map(delayed(self._test_one_ex), input_batches_per_pool))

    def _test_one_ex(self, inputs):
        """
            This function performs testing on one example.
        :param inputs:
                    [summaries, files[start: end], enc_inp, enc_pad_mask, enc_doc_mask,
                     enc_inp_ext_vocab, ext_vocabs, max_oov_size, iteration]
        :return:
        """
        iteration = inputs[0][0]
        input_batches = inputs[1:]
        # Inputs for running beam search on one example.
        beam_search_inputs = input_batches[2: 5]
        if FLAGS.use_pgen:
            beam_search_inputs += [input_batches[5], input_batches[7]]

        best_hyp = run_beam_search_hier(
            beam_search_inputs, self.run_encoder, self.decode_one_step, self._vocab
        )
        pred_ids = best_hyp.tokens                  # Shape: T_dec * [1 x]
        pred_ids = np.stack(pred_ids, axis=0)       # Shape: T_dec x
        pred_ids = pred_ids[np.newaxis, :]          # Shape: 1 x T_dec

        # Writing the outputs.
        if FLAGS.use_pgen:
            self._write_outputs(iteration, pred_ids, input_batches[0], input_batches[1], input_batches[6])
        else:
            self._write_outputs(iteration, pred_ids, input_batches[0], input_batches[1])

    def _get_feed_dict(self, split):
        """
            Returns a feed_dict assigning data batches to the following placeholders:
                [enc_in, enc_pad_mask, enc_doc_mask, dec_in, labels, dec_pad_mask, enc_inp_ext_vocab, max_oov_size]
        :param split:
        :return:
        """

        if split == "train":
            input_batches = self._data.get_batch(
                FLAGS.batch_size, split="train", permutate=FLAGS.permutate, chunk=FLAGS.chunk
            )

        elif split == "val":
            input_batches = self._data.get_batch(
                FLAGS.val_batch_size, split="val", permutate=FLAGS.permutate, chunk=FLAGS.chunk
            )

        else:
            raise ValueError("split should be either train/val!! \n")

        feed_dict = {
            self._enc_in: input_batches[0],
            self._enc_pad_mask: input_batches[1],
            self._enc_doc_mask: input_batches[2],
            self._dec_in: input_batches[3],
            self._dec_out: input_batches[4],
            self._dec_pad_mask: input_batches[5]
        }

        if FLAGS.use_pgen:
            feed_dict[self._enc_in_ext_vocab] = input_batches[6]
            feed_dict[self._max_oov_size] = input_batches[7]

        return feed_dict

    @staticmethod
    def make_html_safe(s):
        """
            Replace any angled brackets in string s to avoid interfering with HTML attention visualizer.
        :param s: Input string.
        :return:
        """
        s.replace("<", "&lt")
        s.replace(">", "&gt")

        return s

    def _write_file(self, pred, pred_name, gt, gt_name, ext_vocab=None):
        """
            This function writes tokens in vals to a .txt file with given name.
        :param pred: Predicted ID's. Shape: 1 x T
        :param pred_name: Name of the file in which predictions will be written.
        :param gt: Ground truth summary, a list of sentences (strings).
        :param gt_name: Name of the file in which GTs will be written.
        :param ext_vocab: Extended vocabulary for each example. [ext_words x]
        :return: _pred, _gt files will be created for ROUGE evaluation.
        """
        # Writing predictions.
        vsize = self._vocab.size()

        # Removing the [START] token.
        pred = pred[1:]

        # Converting words to ID's
        pred_words = []
        for t in pred:
            try:
                pred_words.append(self._vocab.id2word(t))
            except ValueError:
                pred_words.append(ext_vocab[t - vsize])

        # Considering tokens only till STOP token.
        try:
            stop_idx = pred_words.index(params.STOP_DECODING)
        except ValueError:
            stop_idx = len(pred_words)
        pred_words = pred_words[: stop_idx]

        # Creating sentences out of the predicted sequence.
        pred_sents = []
        while pred_words:
            try:
                period_idx = pred_words.index(".")
            except ValueError:
                period_idx = len(pred_words)

            # Append the sentence.
            sent = pred_words[: period_idx + 1]
            pred_sents.append(" ".join(sent))

            # Consider the remaining words now.
            pred_words = pred_words[period_idx + 1:]

        # Making HTML safe.
        pred_sents = [self.make_html_safe(s) for s in pred_sents]
        gt_sents = [self.make_html_safe(s) for s in gt]

        # Writing predicted sentences.
        f = open(pred_name, 'w', encoding='utf-8')
        for i, sent in enumerate(pred_sents):
            f.write(sent) if i == len(pred_sents) - 1 else f.write(sent + "\n")
        f.close()

        # Writing GT sentences.
        f = open(gt_name, 'w', encoding='utf-8')
        for i, sent in enumerate(gt_sents):
            f.write(sent) if i == len(gt_sents) - 1 else f.write(sent + "\n")
        f.close()

    def _write_outputs(self, index, preds, gts, files, ext_vocabs=None):
        """
            This function writes the input files
        :param index: Number of the test example.
        :param preds: The predictions. Shape: B x T
        :param: gts: The ground truths. Shape: B x T
        :param files: The names of the files.
        :param ext_vocabs: Extended vocabularies for each example, Shape: B * [ext_words x]
        :return: Saves the predictions and GT's in a .txt format.
        """
        for i in range(len(files)):
            file, pred, gt = files[i], preds[i], gts[i]
            name_pred = FLAGS.PathToResults + 'predictions/' + '%06d_pred.txt' % index
            name_gt = FLAGS.PathToResults + 'groundtruths/' + '%06d_gt.txt' % index

            ext_vocab = None
            if FLAGS.use_pgen:
                ext_vocab = ext_vocabs[i]

            self._write_file(pred, name_pred, gt, name_gt, ext_vocab)

    def _single_model(self):
        # with tf.device('/gpu'):
        placeholders = self._create_placeholders()
        self._crossentropy_loss, self._samples = self._forward(placeholders)
        sgd_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)                     # Optimizer
        self._train_op = sgd_solver.minimize(
            self._crossentropy_loss, global_step=self._global_step, name='train_op')

        self._saver = tf.train.Saver(var_list=tf.global_variables())                    # Saver.
        self._init = tf.global_variables_initializer()                                  # Initializer.
        self._sess = tf.Session(config=self._config)                                    # Session.

        # Train and validation summary writers.
        if FLAGS.mode == 'train':
            self._create_writers()

        self._global_vars = tf.global_variables()
        # print('No. of variables = {}\n'.format(len(tf.trainable_variables())))
        # print(tf.trainable_variables())
        # no_params = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
        # print('No. of params = {:d}'.format(int(no_params)))

    def _parallel_model(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            placeholders = self._create_placeholders()

            # Splitting the placeholders for each GPU.
            placeholders_per_gpu = [[] for _ in range(self._num_gpus)]
            for placeholder in placeholders:
                splits = tf.split(placeholder, num_or_size_splits=self._num_gpus, axis=0)
                for i, split in enumerate(splits):
                    placeholders_per_gpu[i].append(split)

            sgd_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)         # Optimizer
            tower_grads = []                                                    # Gradients calculated in each tower
            with tf.variable_scope(tf.get_variable_scope()):
                all_losses = []
                all_samples = []
                for i in range(self._num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i):
                            crossentropy_loss, samples = self._forward(placeholders_per_gpu[i])

                            # Gathering samples from all GPUs.
                            all_samples.append(samples)

                            tf.get_variable_scope().reuse_variables()

                            grads = sgd_solver.compute_gradients(crossentropy_loss)
                            tower_grads.append(grads)

                            # Updating the losses
                            all_losses.append(crossentropy_loss)

                self._crossentropy_loss = tf.add_n(all_losses) / self._num_gpus

                # Samples from all the GPUs.
                if FLAGS.rouge_summary:
                    self._samples = tf.concat(all_samples, axis=0)
                else:
                    self._samples = []

            # Synchronization Point
            gradients, variables = zip(*average_gradients(tower_grads))
            gradients, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)

            # Summary for the global norm
            tf.summary.scalar('global_norm', global_norm)

            # Histograms for gradients.
            for grad, var in zip(gradients, variables):
                # if (grad is not None) and ("word_embedding" in var.name):
                tf.summary.histogram(var.op.name + '/gradients', grad)

            # Histograms for variables.
            for var in tf.trainable_variables():
                # if "word_embedding" in var.name:
                tf.summary.histogram(var.op.name, var)

            self._train_op = sgd_solver.apply_gradients(
                zip(gradients, variables), global_step=self._global_step)
            self._saver = tf.train.Saver(var_list=tf.global_variables(),
                                         max_to_keep=None)                    # Saver.
            self._init = tf.global_variables_initializer()                    # Initializer.
            self._sess = tf.Session(config=self._config)                      # Session.

            # Train and validation summary writers.
            if FLAGS.mode == 'train':
                self._create_writers()

            self._global_vars = tf.global_variables()
            # print('No. of variables = {}\n'.format(len(tf.trainable_variables())))
            # print(tf.trainable_variables())
            # no_params = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
            # print('No. of params = {:d}'.format(int(no_params)))
