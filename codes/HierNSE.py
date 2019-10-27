# -*- coding: utf-8 -*-
__author__ = "Rajeev Bhatt Ambati"

import tensorflow as tf
tf.set_random_seed(2018)


def create_rnn_cell(rnn_size, num_layers, scope):
    """
        This function creates and returns an RNN cell.
    :param rnn_size: Size of the hidden state.
    :param num_layers: No. of layers if using a multi_rnn cell.
    :param scope: scope for the RNN variables.
    :return: returns the RNN cell with the necessary specifications.
    """
    with tf.variable_scope(scope):
        layers = []
        for _ in range(num_layers):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=rnn_size, reuse=tf.AUTO_REUSE)
            layers.append(cell)

        if num_layers > 1:
            return tf.nn.rnn_cell.MultiRNNCell(layers)
        else:
            return layers[0]


class HierNSE:
    """
        This is a Hierarchical Neural Semantic Encoder class.
    """

    def __init__(self, batch_size, dim, dense_init, mode='train', use_comp_lstm=False, num_layers=1):
        """
        :param batch_size: No. of examples in a batch of data.
        :param dim: Dimension of the memories. (Same as the dimension of wordvecs).
        :param dense_init: Dense kernel initializer.
        :param mode: 'train/val/test' mode.
        :param use_comp_lstm: Flag if LSTM should be used for compose function or MLP.
        :param num_layers: Number of layers in the RNN cell.
        """
        self._batch_size, self._dim = batch_size, dim

        self._dense_init = dense_init   # Initializer.

        self._mode = mode
        self._use_comp_lstm = use_comp_lstm

        self._read_scope = 'read'
        self._write_scope = 'write'
        self._comp_scope = 'compose'

        # Read LSTM
        self._read_lstm = create_rnn_cell(self._dim, num_layers, self._read_scope)

        # Compose LSTM
        if self._use_comp_lstm:
            self._comp_lstm = create_rnn_cell(3*self._dim, num_layers, self._comp_scope)

        # Write LSTM
        self._write_lstm = create_rnn_cell(self._dim, num_layers, self._write_scope)

    def read(self, x_t, state=None):
        """
            This is the read function.
        :param x_t: input sequence x, Shape: B x D.
        :param state: Previous hidden state of read LSTM.
        :return: r_t: Outputs of the read LSTM, Shape: B x D
        """
        with tf.variable_scope(self._read_scope, reuse=tf.AUTO_REUSE):
            r_t, state = self._read_lstm(x_t, state)

        return r_t, state

    def compose(self, r_t, zs_t, ms_t, zd_t, md_t, state=None):
        """
            This is the compose function.
        :param r_t: Read from the input x_t, Shape: B x D.
        :param zs_t: Attention distribution over sentence memory, Shape: B x T.
        :param ms_t: Sentence memory at time step 't', Shape: B x T x D.
        :param zd_t: Attention distribution over document memory, Shape: B x S.
        :param md_t: Document memory at time step 't', Shape: B x S x D.
        :param state: Previous hidden state of compose LSTM.
        :return: ms_rt: Retrieved sentence memory, Shape: B x D.
                 md_rt: Retrieved document memory, Shape: B x D.
                 c_t: The composed vector, Shape: B x D.
                 state: Hidden state of compose LSTM after previous time step if using it.
        """
        # Retrieved memories.
        ms_rt = tf.squeeze(tf.matmul(tf.expand_dims(zs_t, axis=1), ms_t), axis=1)   # Sentence memory, Shape: B x D.
        md_rt = tf.squeeze(tf.matmul(tf.expand_dims(zd_t, axis=1), md_t), axis=1)   # Document memory, Shape: B x D.

        with tf.variable_scope(self._comp_scope, reuse=tf.AUTO_REUSE):
            r_m_t = tf.concat([r_t, ms_rt, md_rt], axis=-1)                         # B x (3*D).

            # Compose LSTM
            if self._use_comp_lstm:
                r_m_t, state = self._comp_lstm(r_m_t, state)

            # Dense layer to reduce size from 3*D to D.
            c_t = tf.layers.dense(inputs=r_m_t,
                                  units=self._dim,
                                  activation=None,
                                  kernel_initializer=self._dense_init,
                                  name='MLP')                                       # Composed vector, Shape: B x D.
            c_t = tf.nn.relu(c_t)                                                   # Activation Function.

        return ms_rt, md_rt, c_t, state

    def write(self, c_t, state=None):
        """
            This function implements the write operation - equation 5 from the paper [1].
        :param c_t: The composed vector, Shape: B x D.
        :param state: Previous hidden state of write LSTM.
        :return: h_t: The write vector,  Shape: B x D
        """
        with tf.variable_scope(self._write_scope, reuse=tf.AUTO_REUSE):
            h_t, state = self._write_lstm(c_t, state)

        return h_t, state

    def attention(self, r_t, m_t, mem_mask, scope="attention"):
        """
            This function computes the attention distribution.
        :param r_t: Read from the input x_t, Shape: B x D.
        :param m_t: Memory at time step 't', Shape: B x T' x D.
                    T' is T for sentence memory and S for document memory.
        :param mem_mask: A mask to indicate the presence of PAD tokens, Shape: B x T'.
        :param scope: Name of the scope.
        :return:
        """
        # Shapes
        batch_size, attn_len, attn_vec_size = m_t.get_shape().as_list()

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Input features.
            with tf.variable_scope("input"):
                input_features = tf.layers.dense(inputs=r_t,
                                                 units=attn_vec_size,
                                                 activation=None,
                                                 kernel_initializer=self._dense_init,
                                                 reuse=tf.AUTO_REUSE,
                                                 name="inp_dense")                      # Shape: B x D.
                input_features = tf.expand_dims(input_features, axis=1)                 # Shape: B x 1 x D.

            # Memory features
            with tf.variable_scope("memory"):
                memory_features = tf.layers.dense(inputs=m_t,
                                                  units=attn_vec_size,
                                                  activation=None,
                                                  kernel_initializer=self._dense_init,
                                                  use_bias=False,
                                                  reuse=tf.AUTO_REUSE,
                                                  name="memory_dense")                  # Shape: B x T x D.

            v = tf.get_variable("v", [attn_vec_size])

            scores = tf.reduce_sum(
                v * tf.tanh(input_features + memory_features), axis=2)                  # Shape: B x T'.
            attn_dist = tf.nn.softmax(scores)                                           # Shape: B x T'

            # Assigning zero probability to the PAD tokens.
            # Re-normalizing the probability distribution to sum to one.
            attn_dist = tf.multiply(attn_dist, mem_mask)
            masked_sums = tf.reduce_sum(attn_dist, axis=1, keepdims=True)               # Shape: B x 1
            attn_dist = tf.truediv(attn_dist, masked_sums)                              # Re-normalization.

        return attn_dist

    @ staticmethod
    def update(zs_t, ms_t, zd_t, md_t, h_t):
        """
            This function updates the sentence and document memories as per the retrieved slots.
        :param zs_t: Retrieved attention distribution over sentence memory, Shape: B x T.
        :param ms_t: Sentence memory at time step 't', Shape: B x T x D.
        :param zd_t: Retrieved attention distribution over document memory, Shape: B x S.
        :param md_t: Document memory at time step 't', Shape: B x S x D.
        :param h_t: Write vector, Shape: B x D.
        :return: new_ms: Updated sentence memory, Shape: B x T x D.
                 new_md: Updated document memory, Shape: B x S x D.
        """

        # Write and erase masks for sentence memories.
        write_mask_s = tf.expand_dims(zs_t, axis=2)                 # Shape: B x T x 1.
        erase_mask_s = tf.ones_like(write_mask_s) - write_mask_s    # Shape: B x T x 1.

        # Write and erase masks for document memories.
        write_mask_d = tf.expand_dims(zd_t, axis=2)                 # Shape: B x S x 1.
        erase_mask_d = tf.ones_like(write_mask_d) - write_mask_d    # Shape: B x S x 1.

        # Write tensors for sentence and document memories.
        write_tensor_s = tf.expand_dims(h_t, axis=1)                # Shape: B x 1 x D.
        write_tensor_d = tf.expand_dims(h_t, axis=1)                # Shape: B x 1 x D.

        new_ms = tf.add(tf.multiply(ms_t, erase_mask_s), tf.multiply(write_tensor_s, write_mask_s))
        new_md = tf.add(tf.multiply(md_t, erase_mask_d), tf.multiply(write_tensor_d, write_mask_d))

        return new_ms, new_md

    def prob_gen(self, ms_rt, md_rt, h_t, r_t):
        """
            This function calculates the generation probability from the retrieved sentence, document memories, write
            vector and the input read.
        :param ms_rt: Retrieved sentence memory, Shape: B x D.
        :param md_rt: Retrieved document memory, Shape: B x D.
        :param h_t: Write vector, Shape: B x D
        :param r_t: Read vector from the input, Shape: B x D.
        :return: p_gen, Shape: B x 1.
        """
        with tf.variable_scope('pgen', reuse=tf.AUTO_REUSE):
            inp = tf.concat([ms_rt, md_rt, h_t, r_t], axis=1)                    # Shape: B x (4*D).
            p_gen = tf.layers.dense(inp,
                                    units=1,
                                    activation=None,
                                    kernel_initializer=self._dense_init,
                                    name='pgen_dense')                           # Dense Layer, Shape: B x 1.
            p_gen = tf.nn.sigmoid(p_gen)                                         # Sigmoid.

        return p_gen

    def step(self, x_t, mem_masks, prev_state, use_pgen=False):
        """
            This function performs one-step of Hier-NSE.
        :param x_t: Input in the current time-step.
        :param mem_masks: [mem_mask_s, mem_mask_d]: Masks for sentence and document memory respectively
                                                   indicating the presence of PAD tokens, Shape: [B x T, B x S].
        :param prev_state: Internal state of the NSE after the previous time-step.
                [0]: [memory_s, memory_d]: The NSE sentence and document memory
                                           respectively, Shape: [B x T x D, B x S x D].
                [1]: read_state: Hidden state of the Read LSTM, Shape: B x D.
                [2]: write_state: Hidden state of the write LSTM, Shape: B x D.
                [3]: comp_state: Hidden state of the compose LSTM, Shape: B x (3*D).
        :param use_pgen: Flag whether pointer mechanism has to be used.
        :return:
                outputs: The following outputs after the current time step.
                    [0]: write vector: The written vector to NSE memory, Shape: B x D.
                    [1]: p_attn: [zs_t, zd_t] Attention distribution for sentence and
                                              document memories respectively, Shape: [B x T, B x S].
                    Following additional output in pointer generator mode.
                    [2]: p_gen: Generation probability, Shape: B x 1.
                    Following additional output when using coverage.
                    [3]: [curr_cov_s, curr_cov_d]: Updated coverages for sentence and document memory respectively.
                state: The internal state of NSE after the current time-step.
                       [[memory_s, memory_d], read_state, write_state, comp_state].
        """
        # Memory masks
        mem_mask_s, mem_mask_d = mem_masks

        # NSE Internal states.
        prev_comp_state = None
        [prev_memory_s, prev_memory_d], prev_read_state, prev_write_state = prev_state[: 3]
        if self._use_comp_lstm:
            prev_comp_state = prev_state[3]

        if prev_read_state is None:     # zero state of read LSTM for the first time step.
            prev_read_state = self._read_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        if prev_write_state is None:    # zero state of write LSTM for the first time step.
            prev_write_state = self._write_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        if (prev_comp_state is None) and self._use_comp_lstm:   # zero state of compose LSTM for the first time step.
            prev_comp_state = self._comp_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        r_t, curr_read_state = self.read(x_t, prev_read_state)                      # Read step.
        zs_t = self.attention(
            r_t=r_t, m_t=prev_memory_s, mem_mask=mem_mask_s
        )                                                                           # Sentence attention distribution.
        zd_t = self.attention(
            r_t=r_t, m_t=prev_memory_d, mem_mask=mem_mask_d
        )                                                                           # Document attention distribution.

        ms_rt, md_rt, c_t, curr_comp_state = self.compose(
            r_t, zs_t, prev_memory_s, zd_t, prev_memory_d, prev_comp_state
        )                                                                           # Compose step.
        h_t, curr_write_state = self.write(c_t, prev_write_state)                   # Write step.
        curr_memory_s, curr_memory_d = self.update(
            zs_t, prev_memory_s, zd_t, prev_memory_d, h_t
        )                                                                           # Update step.

        curr_state = [[curr_memory_s, curr_memory_d], curr_read_state, curr_write_state]    # Current NSE states.
        if self._use_comp_lstm:
            curr_state.append(curr_comp_state)

        outputs = [h_t, [zs_t, zd_t]]
        if use_pgen:
            p_gen = self.prob_gen(ms_rt, md_rt, h_t, r_t)                           # Generation probability.
            outputs.append(p_gen)

        return outputs, curr_state
