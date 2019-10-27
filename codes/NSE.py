# -*- coding: utf-8 -*-
__author__ = "Rajeev Bhatt Ambati"

import tensorflow as tf
tf.set_random_seed(2019)


def create_rnn_cell(rnn_size, scope):
    """
        This function creates and returns an RNN cell.
    :param rnn_size: Size of the hidden state.
    :param scope: scope for the RNN variables.
    :return: returns the RNN cell with the necessary specifications.
    """
    with tf.variable_scope(scope):
        cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=rnn_size, reuse=tf.AUTO_REUSE)

    return cell


class NSE:
    """
        This is a Neural Semantic Encoder class.
    """

    def __init__(self, batch_size, dim, dense_init, mode='train', use_comp_lstm=False):
        """
        :param batch_size: No. of examples in a batch of data.
        :param dim: Dimension of the memories. (Same as the dimension of wordvecs).
        :param dense_init: Dense kernel initializer.
        :param mode: 'train/val/test' mode.
        :param use_comp_lstm: Flag if LSTM should be used for compose function or MLP.
        """
        self._batch_size, self._dim = batch_size, dim

        self._dense_init = dense_init   # Initializer.

        self._mode = mode
        self._use_comp_lstm = use_comp_lstm

        self._read_scope = 'read'
        self._write_scope = 'write'
        self._comp_scope = 'compose'

        # Read LSTM
        self._read_lstm = create_rnn_cell(self._dim, self._read_scope)

        # Compose LSTM
        if self._use_comp_lstm:
            self._comp_lstm = create_rnn_cell(2*self._dim, self._comp_scope)

        # Write LSTM
        self._write_lstm = create_rnn_cell(self._dim, self._write_scope)

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

    def compose(self, r_t, z_t, m_t, state=None):
        """
            This is the compose function.
        :param r_t: Read from the input x_t, Shape: B x D.
        :param z_t: Attention distribution, Shape: B x T.
        :param m_t: Memory at time step 't', Shape: B x T x D.
        :param state: Previous hidden state of compose LSTM.
        :return: m_rt: Retrieved memory, Shape: B x D
                 c_t: The composed vector, Shape: B x D.
                 state: Hidden state of compose LSTM after previous time step if using it.
        """
        # z_t is repeated across dimension.
        z_t_rep = tf.tile(tf.expand_dims(z_t, axis=-1), multiples=[1, 1, self._dim])  # Shape: B x T x D.
        m_rt = tf.reduce_sum(tf.multiply(z_t_rep, m_t), axis=1)                       # Retrieved memory, Shape: B x D

        with tf.variable_scope(self._comp_scope, reuse=tf.AUTO_REUSE):
            r_m_t = tf.concat([r_t, m_rt], axis=-1)                                   # Shape: B x (2*D)

            # Compose LSTM
            if self._use_comp_lstm:
                r_m_t, state = self._comp_lstm(r_m_t, state)

            # Dense layer to reduce size from (2*D) to D.
            c_t = tf.layers.dense(inputs=r_m_t,
                                  units=self._dim,
                                  activation=None,
                                  kernel_initializer=self._dense_init,
                                  name='MLP')                                         # Composed vector, Shape: B x D
            c_t = tf.nn.relu(c_t)                                                     # Activation function

        return m_rt, c_t, state

    def write(self, c_t, state=None):
        """
            This is the write function.
        :param c_t: The composed vector, Shape: B x D.
        :param state: Previous hidden state of write LSTM.
        :return: h_t: The write vector,  Shape: B x D
        """
        with tf.variable_scope(self._write_scope, reuse=tf.AUTO_REUSE):
            h_t, state = self._write_lstm(c_t, state)

        return h_t, state

    def attention(self, r_t, m_t, mem_mask):
        """
            This function computes the attention distribution.
        :param r_t: Read from the input x_t, Shape: B x D.
        :param m_t: Memory at time step 't', Shape: B x T x D.
        :param mem_mask: A mask to indicate the presence of PAD tokens, Shape: B x T.
        :return:
            attn_dist: The attention distribution at the current time-step, Shape: B x T.
        """
        # Shapes
        attn_len, attn_vec_size = m_t.get_shape().as_list()[1: 3]

        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
            # Input features.
            with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
                input_features = tf.layers.dense(inputs=r_t,
                                                 units=attn_vec_size,
                                                 activation=None,
                                                 kernel_initializer=self._dense_init,
                                                 reuse=tf.AUTO_REUSE,
                                                 name='inp_dense')                      # Shape: B x D.

                input_features = tf.expand_dims(input_features, axis=1)                 # Shape: B x 1 x D.

            # Memory features.
            with tf.variable_scope("memory", reuse=tf.AUTO_REUSE):
                memory_features = tf.layers.dense(inputs=m_t,
                                                  units=attn_vec_size,
                                                  activation=None,
                                                  kernel_initializer=self._dense_init,
                                                  use_bias=False,
                                                  reuse=tf.AUTO_REUSE,
                                                  name="memory_dense")                  # Shape: B x T x D.

            v = tf.get_variable("v", [attn_vec_size])

            scores = tf.reduce_sum(
                v * tf.tanh(input_features + memory_features), axis=2)                  # Shape: B x T.
            attn_dist = tf.nn.softmax(scores)                                           # Shape: B x T.

            # Assigning zero probability to the PAD tokens.
            # Re-normalizing the probability distribution to sum to one.
            attn_dist = tf.multiply(attn_dist, mem_mask)
            masked_sums = tf.reduce_sum(attn_dist, axis=1, keepdims=True)               # Shape: B x 1
            attn_dist = tf.truediv(attn_dist, masked_sums)                              # Re-normalization.

        return attn_dist

    @staticmethod
    def update(z_t, m_t, h_t):
        """
            This function updates the memory with write vectors as per the retrieved slots.
        :param z_t: Retrieved attention distribution over memory, Shape: B x T.
        :param m_t: Memory at current time step 't', Shape: B x T x D.
        :param h_t: Write vector, Shape: B x D.
        :return: new_m: The updated memory, Shape: B x T x D.
        """
        # Write and erase mask for sentence memories.
        write_mask = tf.expand_dims(z_t, axis=2)                # Shape: B x T x 1.
        erase_mask = tf.ones_like(write_mask) - write_mask      # Shape: B x T x 1.

        # Write tensor
        write_tensor = tf.expand_dims(h_t, axis=1)              # Shape: B x 1 x D.

        # Updated memory.
        new_m = tf.add(tf.multiply(m_t, erase_mask), tf.multiply(write_tensor, write_mask))

        return new_m

    def prob_gen(self, m_rt, h_t, r_t):
        """
            This function calculates the generation probability from the retrieved memory, write vector and the
            input read.
        :param m_rt: Retrieved memory, Shape: B x D.
        :param h_t:  Write vector, Shape: B x D.
        :param r_t:  Read vector from the input, Shape: B x D.
        :return: p_gen, Shape: B x 1
        """
        with tf.variable_scope('pgen', reuse=tf.AUTO_REUSE):
            inp = tf.concat([m_rt, h_t, r_t], axis=-1)                      # Shape: B x (3*D)
            p_gen = tf.layers.dense(inp,
                                    units=1,
                                    activation=None,
                                    kernel_initializer=self._dense_init,
                                    name='pgen_dense')                      # Shape: B x 1

            p_gen = tf.nn.sigmoid(p_gen)                                    # Sigmoid.

        return p_gen

    def step(self, x_t, mem_mask, prev_state, use_pgen=False):
        """
            This function performs one-step of NSE.
        :param x_t: Input in the current time-step, Shape: B x D.
        :param mem_mask: Memory mask, Shape: B x T.
        :param prev_state: Internal state of NSE after the previous time step.
                [0] memory: The NSE memory, Shape: B x T x D.
                [1] read_state: Hidden state of the read LSTM, Shape: B x D.
                [2] write_state: Hidden state of the write LSTM, Shape: B x D.
                [3] comp_state: Hidden state of compose LSTM, Shape: B x (2*D).
        :param use_pgen: Flag whether pointer mechanism has to be used.
        :return:
            outputs: The outputs after the current time step.
                [0]: write vector: The written vector to NSE memory, Shape: B x D.
                [1]: p_attn: Attention distribution, Shape: B x T_in.
                Following additional output in pointer generator mode:
                    [2]: p_gen: Generation probability, Shape: B x 1.
            state: Internal state of NSE after current time step.
                [memory, read_state, write_state, comp_state]
        """
        prev_comp_state = None
        prev_memory, prev_read_state, prev_write_state = prev_state[: 3]
        if self._use_comp_lstm:
            prev_comp_state = prev_state[3]

        if prev_read_state is None:     # zero state of read LSTM for the first time step.
            prev_read_state = self._read_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        if prev_write_state is None:    # zero state of write LSTM for the first time step.
            prev_write_state = self._write_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        if (prev_comp_state is None) and self._use_comp_lstm:   # zero state of compose LSTM for the first time step.
            prev_comp_state = self._comp_lstm.zero_state(batch_size=self._batch_size, dtype=tf.float32)

        r_t, curr_read_state = self.read(x_t, prev_read_state)                                  # Read step.
        z_t = self.attention(r_t, prev_memory, mem_mask)                                        # Attention.
        m_rt, c_t, curr_comp_state = self.compose(r_t, z_t, prev_memory, prev_comp_state)       # Compose step.
        h_t, curr_write_state = self.write(c_t, prev_write_state)                               # Write step.
        curr_memory = self.update(z_t, prev_memory, h_t)                                        # Memory update step.

        curr_state = [curr_memory, curr_read_state, curr_write_state]               # Current NSE states.
        if self._use_comp_lstm:
            curr_state.append(curr_comp_state)

        outputs = [h_t, z_t]                                                        # Outputs after the current step.
        if use_pgen:                                                                # Pointer generator mode.
            p_gen = self.prob_gen(m_rt, h_t, r_t)
            outputs.append(p_gen)

        return outputs, curr_state
