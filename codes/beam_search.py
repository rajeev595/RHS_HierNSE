import numpy as np
import tensorflow as tf
import params as params

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """
        This is a class that represents a hypothesis.
    """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens):
        """
        :param tokens: Tokens in this hypothesis.
        :param log_probs: log probabilities of each token in the hypothesis.
        :param state: Internal state of decoder after decoding the last token.
        :param attn_dists: Attention distributions of each token.
        :param p_gens: Generation probability of each token.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens

    def extend(self, token, log_prob, state=None, attn_dist=None, p_gen=None):
        """
            This function extends the hypothesis with the given new token.
        :param token: New decoded token.
        :param log_prob: log probability ot the token.
        :param state: Internal state of decoder after decoding the token.
        :param attn_dist: Attention distribution of the token.
        :param p_gen: Generation probability of each token.
        :return: returns the extended hypothesis
        """
        if state is None:
            state = self.state

        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + log_prob,
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
            p_gens=self.p_gens + [p_gen]
        )

    def latest_token(self):
        """
        :return: This function returns the last token in the hypothesis.
        """
        return self.tokens[-1]

    def avg_log_prob(self):
        """
        :return: This function returns the average of the log probability of hypothesis.
                 Otherwise, longer sequences will have less probability.
        """
        return sum(self.log_probs) / len(self.tokens)


def run_beam_search(inputs, run_encoder, decode_one_step, vocab):
    """
        This function performs the beam search decoding for one example.
    :param inputs: Inputs to the graph for running encoder and decoder.
                   [enc_inp, enc_padding_mask, enc_inp_ext_vocab, max_oov_size] all with shapes Bm x T_in.
    :param run_encoder:
    :param decode_one_step:
    :param vocab:
    :return:
    """
    # Run the encoder and fix the resulted final state for decoding process.
    init_states = run_encoder(inputs[0], inputs[1])

    # Initialize beam size number of hypothesis.
    hyps = [Hypothesis(
        tokens=[vocab.word2id(params.START_DECODING)],
        log_probs=[0.0],
        state=init_states,
        attn_dists=[],
        p_gens=[]
    ) for _ in range(FLAGS.beam_size)]

    results = []    # This will contain finished hypothesis (those have emitted [STOP_DECODING] token.)

    for step in range(FLAGS.bs_dec_steps):

        # Stop decoding if beam_size number of complete hypothesis are emitted.
        if len(results) == FLAGS.beam_size:
            break

        prev_tokens = [h.latest_token() for h in hyps]  # Tokens of all hypothesis from previous time step.
        # Replacing the OOV tokens with UNK tokens to perform the next decoding step.
        prev_tokens = [t if t in range(vocab.size()) else vocab.word2id(params.UNKNOWN_TOKEN) for t in prev_tokens]
        prev_states = [h.state for h in hyps]    # Internal states of decoder.

        # Running one step of decoder.
        step_inputs = [prev_tokens, inputs[1]]
        if FLAGS.use_pgen:
            step_inputs += [inputs[2], inputs[3]]

        outputs = decode_one_step(step_inputs, prev_states)
        topk_ids, topk_log_probs, curr_states = outputs[: 3]

        p_gens = FLAGS.beam_size * [None]
        p_attns = FLAGS.beam_size * [None]
        if FLAGS.use_pgen:
            p_gens, p_attns = outputs[3: 5]

        # Extend each hypothesis with newly decoded predictions.
        all_hyps = []
        num_org_hyps = 1 if step == 0 else len(hyps)  # In the first step, there is only 1 distinct hypothesis.
        for i in range(num_org_hyps):
            hyp, curr_state, p_gen, p_attn = hyps[i], curr_states[i], p_gens[i], p_attns[i]
            for j in range(2 * FLAGS.beam_size):
                new_hyp = hyp.extend(token=topk_ids[i][j],
                                     log_prob=topk_log_probs[i][j],
                                     state=curr_state,
                                     attn_dist=p_attn,
                                     p_gen=p_gen)
                all_hyps.append(new_hyp)

        hyps = []   # Top Bm hypothesis.
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.word2id(params.STOP_DECODING):   # If hypothesis ended.
                # Collect this sequence only if it is long enough.
                if step >= FLAGS.min_dec_len:
                    results.append(h)
            else:   # Use this hypothesis for next decoding step if not ended.
                hyps.append(h)

            # Stop if beam size is reached.
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                break

    # If no complete hypothesis were collected, add all current hypothesis to the results.
    if len(results) == 0:
        results = hyps

    results = sort_hyps(results)    # Return best hypothesis in the final beam.

    return results[0]


def sort_hyps(hyps):
    """
        This function sorts the given hypothesis based on its probability.
    :param hyps: Input hypotheses.
    :return:
    """
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)


def run_beam_search_hier(inputs, run_encoder, decode_one_step, vocab):
    """
        This function performs the beam search decoding for one example.
    :param inputs: Inputs to the graph for running encoder and decoder.
                   [enc_inp, enc_pad_mask, enc_doc_mask, enc_inp_ext_vocab, max_oov_size] all with shapes Bm x T_in.
    :param run_encoder:
    :param decode_one_step:
    :param vocab:
    :return:
    """
    # Run the encoder and fix the resulted final state for decoding process.
    init_states = run_encoder(inputs[: 3])

    # Initialize beam size number of hypothesis.
    hyps = [Hypothesis(
        tokens=[vocab.word2id(params.START_DECODING)],
        log_probs=[0.0],
        state=init_states,
        attn_dists=[],
        p_gens=[]
    ) for _ in range(FLAGS.beam_size)]

    results = []    # This will contain finished hypothesis (those have emitted [STOP_DECODING] token.)

    for step in range(FLAGS.bs_dec_steps):

        # Stop decoding if beam_size number of complete hypothesis are emitted.
        if len(results) == FLAGS.beam_size:
            break

        prev_tokens = [h.latest_token() for h in hyps]  # Tokens of all hypothesis from previous time step.
        # Replacing the OOV tokens with UNK tokens to perform the next decoding step.
        prev_tokens = [t if t in range(vocab.size()) else vocab.word2id(params.UNKNOWN_TOKEN) for t in prev_tokens]
        prev_states = [h.state for h in hyps]    # Internal states of decoder.

        # Preparing inputs for the decoder.
        # inputs[1] = np.reshape(inputs[1], [-1, FLAGS.max_enc_sent * FLAGS.max_enc_steps_per_sent])  # B x T_enc.
        # Running one step of decoder.
        step_inputs = [prev_tokens] + inputs[1: 3]
        if FLAGS.use_pgen:
            step_inputs += [inputs[3], inputs[4]]

        outputs = decode_one_step(step_inputs, prev_states)
        topk_ids, topk_log_probs, curr_states = outputs[: 3]

        p_gens = FLAGS.beam_size * [None]
        p_attns = FLAGS.beam_size * [None]
        if FLAGS.use_pgen:
            p_gens, p_attns = outputs[3: 5]

        # Extend each hypothesis with newly decoded predictions.
        all_hyps = []
        num_org_hyps = 1 if step == 0 else len(hyps)  # In the first step, there is only 1 distinct hypothesis.
        for i in range(num_org_hyps):
            hyp, curr_state, p_gen, p_attn = hyps[i], curr_states[i], p_gens[i], p_attns[i]
            for j in range(2 * FLAGS.beam_size):
                new_hyp = hyp.extend(token=topk_ids[i][j],
                                     log_prob=topk_log_probs[i][j],
                                     state=curr_state,
                                     attn_dist=p_attn,
                                     p_gen=p_gen)
                all_hyps.append(new_hyp)

        hyps = []   # Top Bm hypothesis.
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.word2id(params.STOP_DECODING):   # If hypothesis ended.
                # Collect this sequence only if it is long enough.
                if step >= FLAGS.min_dec_len:
                    results.append(h)
            else:                                           # Use this hypothesis for next decoding step if not ended.
                hyps.append(h)

            # Stop if beam size is reached.
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                break

    # If no complete hypothesis were collected, add all current hypothesis to the results.
    if len(results) == 0:
        results = hyps

    results = sort_hyps(results)    # Return best hypothesis in the final beam.

    return results[0]
