import numpy as np


def rouge_l_fscore_batch(hypothesis, references, end_id, lens, all_steps=False):
    """
        ROUGE scores computation between labels and predictions.
        This is an approximate ROUGE scoring method since we do not glue word pieces
        or decode the ids and tokenize the output.
    :param hypothesis: (predictions) tensor, model predictions (batch_size, <=max_dec_steps)
    :param references: (labels) tensor, gold output. (batch_size, max_dec_steps)
    :param end_id: End of sequence ID.
    :param lens: Lengths of the sequences excluding the PAD tokens, Shape: (batch_size).
    :param all_steps: Whether to calculate rewards for all time-steps.
    :return: rouge_l_fscore: approx rouge-l f1 score, Shape: (batch_size).
    """

    if all_steps:
        batch_fscore = rouge_l_sentence_level_all_batch(hypothesis, references, end_id, lens)       # Shape: B x T.
    else:
        batch_fscore = rouge_l_sentence_level_final_batch(hypothesis, references, end_id, lens)     # Shape: B x .

    return batch_fscore


def infer_length(seq, end_id):
    """
        This function is used to calculate the length of given sequence based on the end ID.
    :param seq: Input sequence, Shape: B x T.
    :param end_id: End of sequence ID.
    :return:
    """
    batch_size = seq.shape[0]
    is_end = np.equal(seq, end_id).astype(np.int32)                             # Shape: B x T.

    # Avoiding the zero length case.
    front_zeros = np.zeros([batch_size, 1], dtype=np.int32)                     # Shape: B x 1.
    is_end = np.concatenate([front_zeros, is_end], axis=1)                      # Shape: B x (T + 1).
    is_end = is_end[:, : -1]                                                    # Shape: B x T.

    count_end = np.cumsum(is_end, axis=1)
    lengths = np.sum(np.equal(count_end, 0).astype(np.int32), axis=1)

    return lengths


def rouge_l_sentence_level_final_batch(eval_sentences, ref_sentences, end_id=None, lens=None):
    """
        Computes ROUGE-L (sentence level) of two collections of sentences.
        Source: https://www.microsoft.com/en-us/research/publication/
        rouge-a-package-for-automatic-evaluation-of-summaries/
        Calculated according to:
        R_lcs = LCS(X,Y)/m
        P_lcs = LCS(X,Y)/n
        F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
        where:
        X = reference summary
        Y = Candidate summary
        m = length of reference summary
        n = length of candidate summary
    :param eval_sentences: The sentences that have been picked by the summarizer, Shape: (batch_size, m)
    :param ref_sentences: The sentences from the reference set, Shape: (batch_size, n)
    :param end_id: End of sentence ID.
    :param lens: Lengths of the sequences excluding PAD tokens.
    :return: F_lcs for all sentences in the batch, Shape: (batch_size,)
    """
    if lens is not None:
        n, m = lens, lens
    else:
        n, m = infer_length(eval_sentences, end_id), infer_length(ref_sentences, end_id)
    lcs = _len_lcs_batch(eval_sentences, ref_sentences, n, m)
    return np.array(_f_lcs_batch(lcs, n, m)).astype(np.float32)


def rouge_l_sentence_level_all_batch(eval_sentences, ref_sentences, end_id, lens):
    """
        Computes ROUGE-L (sentence level) of two collections of sentences.
        Source: https://www.microsoft.com/en-us/research/publication/
        rouge-a-package-for-automatic-evaluation-of-summaries/
        Calculated according to:
        R_lcs = LCS(X,Y)/m
        P_lcs = LCS(X,Y)/n
        F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
        where:
        X = reference summary
        Y = Candidate summary
        m = length of reference summary
        n = length of candidate summary
    :param eval_sentences: The sentences that have been picked by the summarizer, Shape: (batch_size, T).
    :param ref_sentences: The sentences from the reference set, Shape: (batch_size, T).
    :param end_id: End of sentence ID.
    :param lens: length of the sequences, Shape: (batch_size).
    :return: F_lcs: Shape: B x T.
    """
    if lens is not None:
        m = lens
    else:
        m = infer_length(ref_sentences, end_id)

    batch_size, steps = eval_sentences.shape

    n = np.tile(np.arange(1, steps + 1), batch_size)                                # Shape: (B*T) x.
    m = np.tile(m, steps)                                                           # Shape: (B*T) x.

    # Calculate F1 scores.
    lcs = _len_lcs_batch(eval_sentences, ref_sentences, n, m, True)                 # Shape: B x T.
    lcs = np.reshape(lcs, [-1])                                                     # Shape: (B*T) x.
    f1_scores_all_steps = _f_lcs_batch(lcs, n, m)                                   # Shape: (B*T,)
    f1_scores = np.reshape(f1_scores_all_steps, (batch_size, steps))                # Shape: B x T.

    return f1_scores


def _len_lcs_batch(x, y, n, m, all_steps=False):
    """
        Returns the length of Longest Common Sub-sequence between two steps.
    :param x: sequence of words, Shape: (batch_size, n).
    :param y: sequence of words, Shape: (batch_size, m).
    :param n: Lengths of the sequences in X, Shape: (batch_size,).
    :param m: Lengths of the sequences in Y, Shape: (batch_size,).
    :param all_steps: Whether to output LCS of all time-steps.
    :return: Lengths of LCS between a batch of x and y, Shape: (batch_size,) / (batch_size, T)
    """
    table = _lcs_batch(x, y)    # Shape: batch_size x len x len.
    len_lcs = []
    for i in range(x.shape[0]):
        if all_steps:
            len_lcs.append(table[i, 1:, m[i]])
        else:
            len_lcs.append(table[i, n[i], m[i]])

    return np.array(len_lcs)


def _lcs_batch(x, y):
    """
        Computes the length of LCS between two seqs.
    :param x: collection of words, (batch_size, n).
    :param y: collection of words, (batch_size, m).
    :return:
    """
    batch_size, n = x.shape
    m = y.shape[1]

    table = np.ndarray(shape=[batch_size, n + 1, m + 1], dtype=np.int32)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[:, i, j] = 0
            else:
                true_indcs = np.argwhere(x[:, i - 1] == y[:, j - 1])
                false_indcs = np.argwhere(x[:, i - 1] != y[:, j - 1])
                table[true_indcs, i, j] = table[true_indcs, i - 1, j - 1] + 1
                table[false_indcs, i, j] = np.maximum(table[false_indcs, i - 1, j], table[false_indcs, i, j - 1])

    return table


def _f_lcs_batch(llcs, n, m):
    """
        Computes the LCS-based F-measure score.
    :param llcs: lengths of LCS, Shape: (batch_size,)
    :param n: number of words in candidate summary, Shape: (batch_size,)
    :param m: number of words in reference summary, Shape: (batch_size,)
    :return:
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)

    return f_lcs
