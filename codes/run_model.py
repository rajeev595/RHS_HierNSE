# -*- coding: utf-8 -*-
# Latest Command.
# python run_model.py --sample=True --restore_type="all" --enc_steps=24 --dec_steps=12 --max_enc_sent=6
# --max_enc_steps_per_sent=4 --max_dec_sent=4 --max_dec_steps_per_sent=3 --prob_dist="softmax" --rnn="LSTM"
# --skip_rnn=False --use_comp_lstm=True --use_pos_emb=False --use_pgen=True --use_enc_coverage=True
# --use_dec_coverage=False --use_reweight=False --vocab_size=20 --num_epochs=10 --val_every=1 --model="hier2"
# --attn_type="ffn" --scaled=True --num_heads=1 --num_high=4 --num_high_heads=5 --num_high_iters=5 --mode="test"
# --batch_size=3 --val_batch_size=3 --PathToCheckpoint="./my_nse_net/model_epoch10"

__author__ = "Rajeev Bhatt Ambati"
import tensorflow as tf
from utils import Vocab, DataGenerator, DataGeneratorHier, eval_model
from model import SummarizationModel
from model_hier import SummarizationModelHier
from model_hier_sc import SummarizationModelHierSC
import os
import random
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

# Parameters obtained from the input arguments.

# Paths.
tf.app.flags.DEFINE_string('PathToDataset', '../data/', 'Path to the datasets.')
tf.app.flags.DEFINE_string('PathToGlove', '../data/glove.840B.300d.w2v.txt',
                           'Path to the pre-trained GloVe Word vectors.')
tf.app.flags.DEFINE_string('PathToVocab', '../data/vocab.txt', 'Path to a vocabulary file if already stored. '
                                                               'Otherwise, a new vocabulary file will be stored here.')
tf.app.flags.DEFINE_string('PathToLookups', '../data/lookups.pkl', 'Path to the lookup tables .pkl file if already'
                                                                   'stored. Otherwise, a new vocabulary file will be'
                                                                   'stored here.')
tf.app.flags.DEFINE_string('PathToResults', '../results/', 'Path to the test results.')
tf.app.flags.DEFINE_string('PathToCheckpoint', './my_nse_net/hier_v14/model_epoch10',
                           'Trained model will be stored here.')
tf.app.flags.DEFINE_boolean('sample', False, 'Sample debugging the code or training for long.')
tf.app.flags.DEFINE_bool('permutate', False, 'Whether to permutate or truncate sequences.')
tf.app.flags.DEFINE_boolean('chunk', True, 'Whether to use chunks in place of sentences to use maximum effective '
                                           'sequence lengths possible')
tf.app.flags.DEFINE_string('PathToTB', 'log/', 'Tensorboard visualization directory.')
tf.app.flags.DEFINE_boolean('restore_checkpoint', True, 'Boolean describing whether training has to be restored from '
                                                        'a checkpoint or start fresh.')
tf.app.flags.DEFINE_string('restore_type', "all", 'String describing whether coverage/momentum parameters has to be'
                                                  'restored or initialized.')

# Plain model inputs.
tf.app.flags.DEFINE_integer('enc_steps', 400, 'No. of time steps in the encoder.')
tf.app.flags.DEFINE_integer('dec_steps', 100, 'No. of time steps in the decoder.')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'Max. no of time steps in the decoder during decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum no. of tokens in a complete hypothesis while decoding.')

# Hier model inputs.
tf.app.flags.DEFINE_integer('max_enc_sent', 20, 'Max. no. of sentences in the encoder.')
tf.app.flags.DEFINE_integer('max_enc_steps_per_sent', 20, 'Max. no. of tokens per a sentence in the encoder.')

# Common flags.
tf.app.flags.DEFINE_integer('num_layers', 1, "No. of layers in each LSTM.")
tf.app.flags.DEFINE_boolean('use_comp_lstm', True, 'Whether to use an LSTM for compose function.')
tf.app.flags.DEFINE_boolean('use_pgen', True, 'Flag whether pointer mechanism should be used.')
tf.app.flags.DEFINE_boolean('use_pretrained', True, 'Flag whether pre-trained word-vectors has to be used.')

# Common sizes.
tf.app.flags.DEFINE_integer('batch_size', 60, 'No. of examples in a batch of training data.')
tf.app.flags.DEFINE_integer('val_batch_size', 60, 'No. of examples in a batch of validation data.')
tf.app.flags.DEFINE_integer('beam_size', 5, 'Beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of the vocabulary.')
tf.app.flags.DEFINE_integer('dim', 300, 'Dimension of the word embedding, it should be the dimension of the '
                                        'pre-trained word vectors used.')

# Common optimizer flags.
tf.app.flags.DEFINE_boolean('use_entropy', True, 'Whether to use entropy of sampling in loss.')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'Maximum gradient norm when gradient clipping.')
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate.')

# Common training flags.
tf.app.flags.DEFINE_boolean('rouge_summary', True, 'A flag whether ROUGE has to be included in the summary.')
tf.app.flags.DEFINE_integer('num_epochs', 20, 'No. of epochs to train the model.')
tf.app.flags.DEFINE_integer('summary_every', 1, 'Write training summaries every few iterations.')
tf.app.flags.DEFINE_integer('val_every', 4, 'No. of training epochs after which model should be validated.')
tf.app.flags.DEFINE_string('mode', 'test', 'train/test mode.')
tf.app.flags.DEFINE_string('model', 'hier', 'Which model to train: plain/hier/rlhier.')
tf.app.flags.DEFINE_list('GPUs', [0], 'GPU ids to be used.')
tf.app.flags.DEFINE_integer('num_pools', 5, 'No. of pools per GPU.')

# Beam search decoding flags.
tf.app.flags.DEFINE_integer('bs_enc_steps', 400, 'No. of time steps in the encoder.')
tf.app.flags.DEFINE_integer('bs_dec_steps', 100, 'No. of time steps in the decoder.')

# Hier model inputs.
tf.app.flags.DEFINE_integer('bs_enc_sent', 20, 'Max. no. of sentences in the encoder.')
tf.app.flags.DEFINE_integer('bs_enc_steps_per_sent', 20, 'Max. no. of tokens per a sentence in the encoder.')

# Self critic policy gradients model.
tf.app.flags.DEFINE_boolean('use_self_critic', False, 'Flag whether to use self critical model.')
tf.app.flags.DEFINE_boolean('teacher_forcing', False, 'Flag whether to use teacher-forcing in greedy mode.')
tf.app.flags.DEFINE_integer('num_samples', 1, 'No. of samples')
tf.app.flags.DEFINE_boolean('use_discounted_rewards', False, 'Flag whether discounted rewards has to be used.')
tf.app.flags.DEFINE_boolean('use_intermediate_rewards', False, 'Flag whether intermediate rewards has to be used.')
tf.app.flags.DEFINE_float('gamma', 0.99, 'Discount Factor')
tf.app.flags.DEFINE_float('eta', 2.5E-5, 'RL/MLE scaling factor.')
tf.app.flags.DEFINE_float('eta1', 0.0, 'Cross-entropy weight.')
tf.app.flags.DEFINE_float('eta2', 0.0, 'RL loss weight.')
tf.app.flags.DEFINE_float('eta3', 1E-4, 'Entropy weight.')


def main(args):
    main_start = time.time()

    tf.set_random_seed(2019)
    random.seed(2019)
    np.random.seed(2019)

    if len(args) != 1:
        raise Exception('Problem with flags: %s' % args)

    # Correcting a few flags for test/eval mode.
    if FLAGS.mode != 'train':
        FLAGS.batch_size = FLAGS.beam_size
        FLAGS.bs_dec_steps = FLAGS.dec_steps

        if FLAGS.model.lower() != "tx":
            FLAGS.dec_steps = 1

    assert FLAGS.mode == 'train' or FLAGS.batch_size == FLAGS.beam_size, \
        "In test mode, batch size should be equal to beam size."

    assert FLAGS.mode == 'train' or FLAGS.dec_steps == 1 or FLAGS.model.lower() == "tx", \
        "In test mode, no. of decoder steps should be one."

    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(gpu_id) for gpu_id in FLAGS.GPUs)

    if not os.path.exists(FLAGS.PathToCheckpoint):
        os.makedirs(FLAGS.PathToCheckpoint)

    if FLAGS.mode == "test" and not os.path.exists(FLAGS.PathToResults):
        os.makedirs(FLAGS.PathToResults)
        os.makedirs(FLAGS.PathToResults + 'predictions')
        os.makedirs(FLAGS.PathToResults + 'groundtruths')

    if FLAGS.mode == 'eval':
        eval_model(FLAGS.PathToResults)
    else:
        start = time.time()
        vocab = Vocab(max_vocab_size=FLAGS.vocab_size, emb_dim=FLAGS.dim, dataset_path=FLAGS.PathToDataset,
                      glove_path=FLAGS.PathToGlove, vocab_path=FLAGS.PathToVocab, lookup_path=FLAGS.PathToLookups)

        if FLAGS.model.lower() == "plain":
            print("Setting up the plain model.\n")
            data = DataGenerator(path_to_dataset=FLAGS.PathToDataset, max_inp_seq_len=FLAGS.enc_steps,
                                 max_out_seq_len=FLAGS.dec_steps, vocab=vocab,
                                 use_pgen=FLAGS.use_pgen, use_sample=FLAGS.sample)
            summarizer = SummarizationModel(vocab, data)

        elif FLAGS.model.lower() == "hier":
            print("Setting up the hier model.\n")
            data = DataGeneratorHier(path_to_dataset=FLAGS.PathToDataset, max_inp_sent=FLAGS.max_enc_sent,
                                     max_inp_tok_per_sent=FLAGS.max_enc_steps_per_sent,
                                     max_out_tok=FLAGS.dec_steps, vocab=vocab,
                                     use_pgen=FLAGS.use_pgen, use_sample=FLAGS.sample)
            summarizer = SummarizationModelHier(vocab, data)

        elif FLAGS.model.lower() == "rlhier":
            print("Setting up the Hier RL model.\n")
            data = DataGeneratorHier(path_to_dataset=FLAGS.PathToDataset, max_inp_sent=FLAGS.max_enc_sent,
                                     max_inp_tok_per_sent=FLAGS.max_enc_steps_per_sent,
                                     max_out_tok=FLAGS.dec_steps, vocab=vocab,
                                     use_pgen=FLAGS.use_pgen, use_sample=FLAGS.sample)
            summarizer = SummarizationModelHierSC(vocab, data)

        else:
            raise ValueError("model flag should be either of plain/hier/bayesian/shared!! \n")

        end = time.time()
        print("Setting up vocab, data and model took {:.2f} sec.".format(end - start))

        summarizer.build_graph()

        if FLAGS.mode == 'train':
            summarizer.train()
        elif FLAGS.mode == "test":
            summarizer.test()
        else:
            raise ValueError("mode should be either train/test!! \n")

        main_end = time.time()
        print("Total time elapsed: %.2f \n" % (main_end - main_start))


if __name__ == '__main__':
    tf.app.run()
