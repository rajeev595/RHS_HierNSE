# -*- coding: utf-8 -*-
__author__ = "Rajeev Bhatt Ambati"
"""
    Acknowledgement: Most of the functions related to vocab object are either inspired/took from the
    Pointer Generator Network repository published by See. et al 2017.
    GitHub link: https://github.com/abisee/pointer-generator/blob/master/data.py
"""

import params
import os
import hashlib
from shutil import copyfile
import math
import collections
import random
from random import shuffle, randint, sample
from gensim.models import KeyedVectors

import pickle
import numpy as np
import tensorflow as tf
# import pyrouge

# Setting-up seeds
random.seed(2019)
np.random.seed(2019)
tf.set_random_seed(2019)


def read_text_file(text_file):

    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())

    return lines


def hashhex(s):
    """
        Returns a heximal formatted SHA1 hash of the input string.
    """
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))

    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def create_set(path_to_data, split):
    if os.path.exists(path_to_data + split):
        return
    else:
        os.makedirs(path_to_data + split)

    url_list = read_text_file(path_to_data + "all_" + split + ".txt")
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s + ".story" for s in url_hashes]

    print("{}:".format(split))
    for i, s in enumerate(story_fnames):
        if os.path.isfile(path_to_data + "cnn_stories_tokenized/" + s):
            src = path_to_data + "cnn_stories_tokenized/" + s

        elif os.path.isfile(path_to_data + "dm_stories_tokenized/" + s):
            src = path_to_data + "dm_stories_tokenized/" + s

        else:
            raise Exception("Story file {} is not found!".format(s))

        trg = path_to_data + split + "/" + s

        copyfile(src, trg)
        print("files done: {}/{}".format(i, len(story_fnames)))


def create_train_val_test(path_to_data="../data/"):
    """
    :param path_to_data: Path to data.
        This function creates train, test and validation sets.
    :return:
    """
    # Train.
    create_set(path_to_data, "train")

    # Validation.
    create_set(path_to_data, "val")

    # Test.
    create_set(path_to_data, "test")


def fix_missing_period(line):
    if "@highlight" in line:
        return line

    if line == "":
        return line

    if line[-1] in params.END_TOKENS:
        return line[: -1] + " " + line[-1]

    return line + " ."


def get_article_summary(story_file, art_format="tokens", sum_format="tokens"):
    # Read the story file.
    f = open(story_file, 'r', encoding='ISO-8859-1')

    # Lowercase everything.
    lines = [line.strip().lower() for line in f.readlines()]

    # Some lines don't have periods in the end. Correct'em.
    lines = [fix_missing_period(line) for line in lines]

    # Separate article and abstract sentences.
    article_lines = []
    highlights = []
    next_is_highlight = False

    for idx, line in enumerate(lines):

        if line == "":  # Empty line.
            continue

        elif line.startswith("@highlight"):
            next_is_highlight = True

        elif next_is_highlight:
            highlights.append(line)

        else:
            article_lines.append(line)

    # Joining all article sentences together into a string.
    article = ' '.join(article_lines)

    # Joining all highlights together into a string including sentence
    # start <s> and end </s>.
    summary = ' '.join(["%s %s %s" % (params.SENTENCE_START, sent,
                                      params.SENTENCE_END) for sent in highlights])

    if art_format == "tokens":

        if sum_format == "tokens":
            return article, summary
        elif sum_format == "sentences":
            return article, highlights
        else:
            raise ValueError("Article format should be either tokens or sentences!! \n")

    elif art_format == "sentences":
        if sum_format == "tokens":
            return article_lines, summary
        elif sum_format == "sentences":
            return article_lines, highlights
        else:
            raise ValueError("Article format should be either tokens or sentences!! \n")
    else:
        raise ValueError("Article format should be either tokens or sentences!! \n")


def article2ids(article_words, vocab):
    """
        This function converts given article words to ID's
    :param article_words: article tokens.
    :param vocab: The vocabulary object used for lookup tables, vocabulary etc.
    :return: The corresponding ID's and a list of OOV tokens.
    """
    ids = []
    oovs = []
    unk_id = vocab.word2id(params.UNKNOWN_TOKEN)
    for word in article_words:
        i = vocab.word2id(word)
        if i == unk_id:  # Out of vocabulary words.
            if word in oovs:
                ids.append(vocab.size() + oovs.index(word))
            else:
                oovs.append(word)
                ids.append(vocab.size() + oovs.index(word))
        else:  # In vocabulary words.
            ids.append(i)

    return ids, oovs


def summary2ids(summary_words, vocab, article_oovs):
    """
        This function converts the given summary words to ID's
    :param summary_words: summary tokens.
    :param vocab: The vocabulary object used for lookup tables, vocabulary etc.
    :param article_oovs: OOV tokens in the input article.
    :return: The corresponding ID's.
    """
    num_sent, sent_len = 0, 0
    if type(summary_words[0]) is list:
        num_sent = len(summary_words)
        sent_len = len(summary_words[0])

        cum_words = []
        for _, sent_words in enumerate(summary_words):
            cum_words += sent_words

        summary_words = cum_words

    ids = []
    unk_id = vocab.word2id(params.UNKNOWN_TOKEN)
    for word in summary_words:
        i = vocab.word2id(word)
        if i == unk_id:  # Out of vocabulary words.
            if word in article_oovs:  # In article OOV words.
                ids.append(vocab.size() + article_oovs.index(word))
            else:  # Both OOV and article OOV words.
                ids.append(unk_id)
        else:  # In vocabulary words.
            ids.append(i)

    if num_sent != 0:
        doc_ids = []
        for i in range(num_sent):
            doc_ids.append(ids[sent_len * i: sent_len * (i + 1)])

        ids = doc_ids

    return ids


class Vocab(object):
    def __init__(self, max_vocab_size, emb_dim=300, dataset_path='../data/', glove_path='../data/glove.840B.300d.txt',
                 vocab_path='../data/vocab.txt', lookup_path='../data/lookup.pkl'):

        self.max_size = max_vocab_size
        self._dim = emb_dim
        self.PathToGloveFile = glove_path
        self.PathToVocabFile = vocab_path
        self.PathToLookups = lookup_path

        create_train_val_test(dataset_path)

        stories = os.listdir(dataset_path + 'train')    # Using only train files for Vocab,

        # All train Stories.
        self._story_files = [
            os.path.join(dataset_path + 'train/', s) for s in stories]

        self.vocab = []  # Vocabulary

        # Create the vocab file.
        self.create_total_vocab()

        # Create the lookup tables.
        self.wvecs = []        # Word vectors.
        self._word_to_id = {}  # word to ID's lookups
        self._id_to_word = {}  # ID to word lookups

        self.create_lookup_tables()

        assert len(self._word_to_id.keys()) == len(self._id_to_word.keys()), "Both lookups should have same size."

    def size(self):
        return len(self.vocab)

    def word2id(self, word):
        """
            This function returns the vocabulary ID for word if it is present. Otherwise, returns the ID
            for the unknown token.
        :param word: input word.
        :return: returns the ID.
        """
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return self._word_to_id[params.UNKNOWN_TOKEN]

    def id2word(self, word_id):
        """
            This function returns the corresponding word for a given vocabulary ID.
        :param word_id: input ID.
        :return:  returns the word.
        """
        if word_id in self._id_to_word:
            return self._id_to_word[word_id]
        else:
            raise ValueError("{} is not a valid ID.\n".format(word_id))

    def create_total_vocab(self):

        if os.path.isfile(self.PathToVocabFile):
            print("Vocab file exists! \n")

            vocab_f = open(self.PathToVocabFile, 'r')
            for line in vocab_f:
                word = line.split()[0]
                self.vocab.append(word)

            return
        else:
            print("Vocab file NOT found!! \n")
            print("Creating a vocab file! \n")

        vocab_counter = collections.Counter()

        for idx, story in enumerate(self._story_files):
            article, summary = get_article_summary(story)

            art_tokens = article.split(' ')                                 # Article tokens.
            sum_tokens = summary.split(' ')                                 # Summary tokens.
            sum_tokens = [t for t in sum_tokens if t not in                 # Removing <s>, </s> tokens.
                          [params.SENTENCE_START, params.SENTENCE_END]]

            assert (params.SENTENCE_START not in sum_tokens) and (params.SENTENCE_END not in sum_tokens), \
                "<s> and </s> shouldn't be present in sum_tokens"

            tokens = art_tokens + sum_tokens
            tokens = [t.strip() for t in tokens]
            tokens = [t for t in tokens if t != '']  # Removing empty tokens.

            vocab_counter.update(tokens)  # Keeping a count of the tokens.

            print("\r{}/{} files read!".format(idx + 1, len(self._story_files)))

        print("\n Writing the vocab file! \n")
        f = open(self.PathToVocabFile, 'w', encoding='utf-8')

        for word, count in vocab_counter.most_common(params.VOCAB_SIZE):
            f.write(word + ' ' + str(count) + '\n')
            self.vocab.append(word)

        f.close()

    def create_small_vocab(self):
        """
            This function selects a few words out of the total vocabulary.
        """

        # Read the vocab file and assign id's to each word till the max_size.
        vocab_f = open(self.PathToVocabFile, 'r')

        for line in vocab_f:
            word = line.split()[0]

            if word in [params.SENTENCE_START, params.SENTENCE_END, params.UNKNOWN_TOKEN,
                        params.PAD_TOKEN, params.START_DECODING, params.STOP_DECODING]:
                raise Exception('<s>, </s>, [UNK], [PAD], [START], \
                                [STOP] shouldn\'t be in the vocab file, but %s is' % word)

            self.vocab.append(word)
            print("\r{}/{} words created!".format(len(self.vocab), self.max_size))

            if len(self.vocab) == self.max_size:
                print("\n Max size of the vocabulary reached! Stopping reading! \n")
                break

    def create_lookup_tables_(self):
        """
            This function creates lookup tables for word vectors, word to IDs
            and ID to words. First max_size words from GloVe that are also found in the small vocab are used
            to create the lookup tables.
        """

        if os.path.isfile(self.PathToLookups):
            print('\n Lookup tables found :) \n')
            f = open(self.PathToLookups, 'rb')
            data = pickle.load(f)

            self._word_to_id = data['word2id']
            self._id_to_word = data['id2word']
            self.wvecs = data['wvecs']
            self.vocab = list(self._word_to_id.keys())

            print('Lookup tables collected for {} tokens.\n'.format(len(self.vocab)))
            return
        else:
            print('\n Lookup files NOT found!! \n')
            print('\n Creating the lookup tables! \n')

        self.create_small_vocab()                                   # Creating a small vocabulary.
        self.wvecs = []                                             # Word vectors.

        glove_f = open(self.PathToGloveFile, 'r', encoding='utf8')
        count = 0

        # [UNK], [PAD], [START] and [STOP] get ids 0, 1, 2, 3.
        for w in [params.UNKNOWN_TOKEN, params.PAD_TOKEN, params.START_DECODING, params.STOP_DECODING]:
            self._word_to_id[w] = count
            self._id_to_word[count] = w
            self.wvecs.append(np.random.uniform(-0.1, 0.1, (self._dim,)).astype(np.float32))
            count += 1

            print("\r Created tables for {}".format(w))

        for line in glove_f:
            vals = line.rstrip().split(' ')
            w = vals[0]
            vec = np.array(vals[1:]).astype(np.float32)

            if w in self.vocab:
                self._word_to_id[w] = count
                self._id_to_word[count] = w
                self.wvecs.append(vec)
                count += 1

                print("\r Created tables for {}".format(w))

            if count == self.max_size:
                print("\r Maximum vocab size reached! \n")
                break

        print("\n Lookup tables created for {} tokens. \n".format(count))

        self.wvecs = np.array(self.wvecs).astype(np.float32)    # Converting to a Numpy array.
        self.vocab = list(self._word_to_id.keys())              # Adjusting the vocabulary to found pre-trained vectors.

        # Saving the lookup tables.
        f = open(self.PathToLookups, 'wb')
        data = {'word2id': self._word_to_id,
                'id2word': self._id_to_word,
                'wvecs': self.wvecs}
        pickle.dump(data, f)

    def create_lookup_tables(self):
        """
            This function creates lookup tables for word vectors, word to IDs
            and ID to words. First max_size words from GloVe that are also found in the small vocab are used
            to create the lookup tables.
        """

        if os.path.isfile(self.PathToLookups):
            print('\n Lookup tables found :) \n')
            f = open(self.PathToLookups, 'rb')
            data = pickle.load(f)

            self._word_to_id = data['word2id']
            self._id_to_word = data['id2word']
            self.wvecs = data['wvecs']
            self.vocab = list(self._word_to_id.keys())

            print('Lookup tables collected for {} tokens.\n'.format(len(self.vocab)))
            return
        else:
            print('\n Lookup files NOT found!! \n')
            print('\n Creating the lookup tables! \n')

        self.wvecs = []                                             # Word vectors.

        word2vec = KeyedVectors.load_word2vec_format(self.PathToGloveFile, binary=False)

        count = 0
        # [UNK], [PAD], [START] and [STOP] get ids 0, 1, 2, 3.
        for w in [params.UNKNOWN_TOKEN, params.PAD_TOKEN, params.START_DECODING, params.STOP_DECODING]:
            self._word_to_id[w] = count
            self._id_to_word[count] = w
            self.wvecs.append(np.random.uniform(-0.1, 0.1, (self._dim,)).astype(np.float32))
            count += 1

            print("\r Created tables for {}".format(w))

        vocab_f = open(self.PathToVocabFile, "r")
        for line in vocab_f:
            word = line.split()[0]

            if word in word2vec:
                self._word_to_id[word] = count
                self._id_to_word[count] = word
                self.wvecs.append(word2vec[word])
                count += 1

                print("\r Created tables for {}".format(word))

            if count == self.max_size:
                print("\r Maximum vocab size reached! \n")
                break

        print("\n Lookup tables created for {} tokens. \n".format(count))

        self.wvecs = np.array(self.wvecs).astype(np.float32)    # Converting to a Numpy array.
        self.vocab = list(self._word_to_id.keys())              # Adjusting the vocabulary to found pre-trained vectors.

        # Saving the lookup tables.
        f = open(self.PathToLookups, 'wb')
        data = {'word2id': self._word_to_id,
                'id2word': self._id_to_word,
                'wvecs': self.wvecs}
        pickle.dump(data, f)


class DataGenerator(object):

    def __init__(self, path_to_dataset, max_inp_seq_len, max_out_seq_len, vocab, use_pgen=False, use_sample=False):
        # Train files.
        train_stories = os.listdir(path_to_dataset + 'train')
        self.train_files = [os.path.join(path_to_dataset + 'train', s) for s in train_stories]
        self.num_train_examples = len(self.train_files)
        shuffle(self.train_files)

        # Validation files.
        val_stories = os.listdir(path_to_dataset + 'val')
        self.val_files = [os.path.join(path_to_dataset + 'val', s) for s in val_stories]
        self.num_val_examples = len(self.val_files)
        shuffle(self.val_files)

        # Test files.
        test_stories = os.listdir(path_to_dataset + 'test')
        self.test_files = [os.path.join(path_to_dataset + 'test', s) for s in test_stories]
        self.num_test_examples = len(self.test_files)
        # shuffle(self.test_files)

        self._max_enc_steps = max_inp_seq_len  # Max. no. of tokens in the input sequence.
        self._max_dec_steps = max_out_seq_len  # Max. no. of tokens in the output sequence.

        self.vocab = vocab  # Vocabulary instance.
        self._use_pgen = use_pgen  # Whether pointer mechanism should be used.

        self._ptr = 0  # Pointer for batching the data.

        if use_sample:
            # **************************** PATCH ************************* #
            self.train_files = self.train_files[:20]
            self.num_train_examples = len(self.train_files)
            self.val_files = self.val_files[:20]
            self.num_val_examples = len(self.val_files)
            self.test_files = self.test_files[:23]
            self.num_test_examples = len(self.test_files)
            # **************************** PATCH ************************* #

        print("Split the data as follows:\n")
        print("\t\t Training: {} examples. \n".format(self.num_train_examples))
        print("\t\t Validation: {} examples. \n".format(self.num_val_examples))
        print("\t\t Test: {} examples. \n".format(self.num_test_examples))

    def get_train_val_batch(self, batch_size, split='train', permutate=False):
        if split == 'train':
            num_examples = self.num_train_examples
            files = self.train_files
        elif split == 'val':
            num_examples = self.num_val_examples
            files = self.val_files
        else:
            raise ValueError("split is neither train nor val. check the function call!")

        enc_inp = np.ndarray(shape=(batch_size, self._max_enc_steps), dtype=np.int32)
        dec_inp = np.ndarray(shape=(batch_size, self._max_dec_steps), dtype=np.int32)
        dec_out = np.ndarray(shape=(batch_size, self._max_dec_steps), dtype=np.int32)

        enc_inp_ext_vocab = None
        max_oov_size = -np.infty
        if self._use_pgen:
            enc_inp_ext_vocab = np.ndarray(shape=(batch_size, self._max_enc_steps), dtype=np.int32)

        # Shuffle files at the start of an epoch.
        if self._ptr == 0:
            shuffle(files)

        # Start and end index for a batch of data.
        start = self._ptr
        end = self._ptr + batch_size
        self._ptr = end

        for i in range(start, end):
            j = i - start  # Index of the example in current batch.
            article, summary = get_article_summary(files[i])
            enc_inp_tokens = article.split(' ')                         # Article tokens.

            # Article Tokens
            if len(enc_inp_tokens) >= self._max_enc_steps:              # Truncate.
                if permutate:
                    indcs = sorted(sample(range(len(enc_inp_tokens)), self._max_enc_steps))
                    enc_inp_tokens = [enc_inp_tokens[i] for i in indcs]
                else:
                    enc_inp_tokens = enc_inp_tokens[: self._max_enc_steps]
            else:                                                       # Pad.
                enc_inp_tokens += (self._max_enc_steps - len(enc_inp_tokens)) * [params.PAD_TOKEN]

            # Encoder Input
            enc_inp_ids = [self.vocab.word2id(w) for w in enc_inp_tokens]  # Word to ID's

            # Summary Tokens
            sum_tokens = summary.split(' ')                             # Summary tokens.
            sum_tokens = [t for t in sum_tokens if t not in             # Removing <s>, </s> tokens.
                          [params.SENTENCE_START, params.SENTENCE_END]]

            if len(sum_tokens) > self._max_dec_steps - 1:               # Truncate.
                sum_tokens = sum_tokens[: self._max_dec_steps - 1]

            # Decoder Input
            dec_inp_tokens = [params.START_DECODING] + sum_tokens
            if len(dec_inp_tokens) < self._max_dec_steps:
                dec_inp_tokens += (self._max_dec_steps - len(dec_inp_tokens)) * [params.PAD_TOKEN]

            # Decoder Output
            dec_out_tokens = sum_tokens + [params.STOP_DECODING]
            dec_out_len = len(dec_out_tokens)
            if dec_out_len < self._max_dec_steps:
                dec_out_tokens += (self._max_dec_steps - dec_out_len) * [params.PAD_TOKEN]

            dec_inp_ids = [self.vocab.word2id(w) for w in dec_inp_tokens]
            dec_out_ids = [self.vocab.word2id(w) for w in dec_out_tokens]

            enc_inp_ids_ext_vocab = None
            if self._use_pgen:
                enc_inp_ids_ext_vocab, article_oovs = article2ids(enc_inp_tokens, self.vocab)
                dec_out_ids = summary2ids(dec_out_tokens, self.vocab, article_oovs)

                if len(article_oovs) > max_oov_size:
                    max_oov_size = len(article_oovs)

            # Appending to the batch of inputs.
            enc_inp[j] = np.array(enc_inp_ids).astype(np.int32)  # Appending to the enc_inp batch.
            dec_inp[j] = np.array(dec_inp_ids).astype(np.int32)  # Appending to the dec_inp batch.
            dec_out[j] = np.array(dec_out_ids).astype(np.int32)  # Appending to the dec_out batch.

            if self._use_pgen:
                enc_inp_ext_vocab[j] = np.array(enc_inp_ids_ext_vocab).astype(np.int32)

        # Resetting the pointer after the last batch
        if self._ptr == num_examples:
            self._ptr = 0

        # Setting the pointer for the last batch
        if self._ptr + batch_size > num_examples:
            self._ptr = num_examples - batch_size

        enc_padding_mask = (enc_inp != self.vocab.word2id(params.PAD_TOKEN)).astype(np.float32)
        dec_padding_mask = (dec_out != self.vocab.word2id(params.PAD_TOKEN)).astype(np.float32)

        batches = [enc_inp, enc_padding_mask, dec_inp, dec_out, dec_padding_mask]
        if self._use_pgen:
            batches += [enc_inp_ext_vocab, max_oov_size]

        return batches

    def get_test_batch(self, batch_size):
        num_examples = self.num_test_examples
        files = self.test_files

        enc_inp = np.ndarray(shape=(batch_size, self._max_enc_steps), dtype=np.int32)

        enc_inp_ext_vocab = None
        max_oov_size = -np.infty
        if self._use_pgen:
            enc_inp_ext_vocab = np.ndarray(shape=(batch_size, self._max_enc_steps), dtype=np.int32)

        summaries = []                                                  # Used in 'test' mode.
        ext_vocabs = []                                                 # Extended vocabularies.

        # # Shuffle files at the start of an epoch.
        # if self._ptr == 0:
        #     shuffle(files)

        # Start and end index for a batch of data.
        start = self._ptr
        end = self._ptr + batch_size
        self._ptr = end

        for i in range(start, end):
            j = i - start                                               # Index of the example in current batch.
            article, summary = get_article_summary(files[i])
            enc_inp_tokens = article.split(' ')

            if len(enc_inp_tokens) >= self._max_enc_steps:              # Truncate.
                enc_inp_tokens = enc_inp_tokens[: self._max_enc_steps]
            else:                                                       # Pad.
                enc_inp_tokens += (self._max_enc_steps - len(enc_inp_tokens)) * [params.PAD_TOKEN]

            # Encoder Input representation in fixed vocabulary.
            enc_inp_ids = [self.vocab.word2id(w) for w in enc_inp_tokens]  # Word to ID's

            # Encoder Input representation in extended vocabulary.
            enc_inp_ids_ext_vocab = None
            article_oovs = None
            if self._use_pgen:
                enc_inp_ids_ext_vocab, article_oovs = article2ids(enc_inp_tokens, self.vocab)

                if len(article_oovs) > max_oov_size:
                    max_oov_size = len(article_oovs)

            # Appending to the input batch.
            enc_inp[j] = np.array(enc_inp_ids).astype(np.int32)

            if self._use_pgen:
                enc_inp_ext_vocab[j] = np.array(enc_inp_ids_ext_vocab).astype(np.int32)
                ext_vocabs.append(article_oovs)

            sum_tokens = summary.split(' ')
            summaries.append(sum_tokens)

        # Resetting the pointer after the last batch
        if self._ptr == num_examples:
            self._ptr = 0

        # Setting the pointer for the last batch
        if self._ptr + batch_size > num_examples:
            self._ptr = num_examples - batch_size

        # Repeat a single input beam size times.
        # enc_inp = np.repeat(enc_inp, repeats=beam_size, axis=0)                         # Shape: Bm x T_in.
        # enc_inp_ext_vocab = np.repeat(enc_inp_ext_vocab, repeats=beam_size, axis=0)     # Shape: Bm x T_in.
        enc_padding_mask = (enc_inp != self.vocab.word2id(params.PAD_TOKEN)).astype(np.float32)    # Shape: B x T_in.

        # Example indices.
        indices = list(range(start, end))
        batches = [indices, summaries, files[start: end], enc_inp, enc_padding_mask]
        if self._use_pgen:
            batches += [enc_inp_ext_vocab, ext_vocabs, max_oov_size]

        return batches

    def get_batch(self, batch_size, split='train', permutate=False):
        if split == 'train' or split == 'val':
            return self.get_train_val_batch(batch_size, split, permutate)
        elif split == 'test':
            return self.get_test_batch(batch_size)
        else:
            raise ValueError('split should be either of train/val/test only!! \n')


class DataGeneratorHier(object):

    def __init__(self, path_to_dataset, max_inp_sent, max_inp_tok_per_sent, max_out_tok, vocab,
                 use_pgen=False, use_sample=False):

        self._max_enc_sent = max_inp_sent                   # Max. no of sentences in the encoder sequence.
        self._max_enc_tok_per_sent = max_inp_tok_per_sent   # Max. no of tokens per sentence in the encoder sequence.
        self._max_enc_tok = self._max_enc_sent * self._max_enc_tok_per_sent
        self._max_dec_tok = max_out_tok                     # Max. no of tokens in the decoder sequence.

        self._vocab = vocab                                 # Vocabulary object.
        self._use_pgen = use_pgen                           # Flag whether pointer-generator mechanism is used.

        self._ptr = 0                                       # Pointer for batching the data.

        # Train files.
        train_stories = os.listdir(path_to_dataset + 'train')
        self._train_files = [os.path.join(path_to_dataset + 'train/', s) for s in train_stories]
        self.num_train_examples = len(self._train_files)
        shuffle(self._train_files)

        # Validation files.
        val_stories = os.listdir(path_to_dataset + 'val')
        self._val_files = [os.path.join(path_to_dataset + 'val/', s) for s in val_stories]
        self.num_val_examples = len(self._val_files)
        shuffle(self._val_files)

        # Test files.
        test_stories = os.listdir(path_to_dataset + 'test')
        self._test_files = [os.path.join(path_to_dataset + 'test/', s) for s in test_stories]
        self.num_test_examples = len(self._test_files)
        # shuffle(self._test_files)

        if use_sample:
            # ***************************** PATCH ***************************** #
            train_idx = randint(0, self.num_train_examples - 20)
            self._train_files = self._train_files[train_idx: train_idx + 20]
            self.num_train_examples = len(self._train_files)

            val_idx = randint(0, self.num_val_examples - 20)
            self._val_files = self._val_files[val_idx: val_idx + 20]
            self.num_val_examples = len(self._val_files)

            test_idx = randint(0, self.num_test_examples - 23)
            self._test_files = self._test_files[test_idx: test_idx + 23]
            self.num_test_examples = len(self._test_files)
            # ***************************** PATCH ***************************** #

        print("Split the data as follows:\n")
        print("\t\t Training: {} examples. \n".format(self.num_train_examples))
        print("\t\t Validation: {} examples. \n".format(self.num_val_examples))
        print("\t\t Test: {} examples. \n".format(self.num_test_examples))

    def get_batch(self, batch_size, split="train", permutate=False, chunk=False):
        if split == "train" or split == "val":
            return self._get_train_val_batch(batch_size, split, permutate, chunk)

        elif split == "test":
            return self._get_test_batch(batch_size, chunk)

        else:
            raise ValueError("Split should be either of train/val/test only!! \n")

    def _get_train_val_batch(self, batch_size, split="train", permutate=False, chunk=False):
        if split == "train":
            num_examples = self.num_train_examples
            files = self._train_files

        elif split == "val":
            num_examples = self.num_val_examples
            files = self._val_files

        else:
            raise ValueError("Split is neither train nor val. Check the function call!")

        enc_inp = np.ndarray(shape=[batch_size, self._max_enc_sent, self._max_enc_tok_per_sent], dtype=np.int32)
        dec_inp = np.ndarray(shape=[batch_size, self._max_dec_tok], dtype=np.int32)
        dec_out = np.ndarray(shape=[batch_size, self._max_dec_tok], dtype=np.int32)

        # Additional inputs in the pointer-generator mode.
        max_oov_size = -np.infty
        enc_inp_ext_vocab = None
        if self._use_pgen:
            enc_inp_ext_vocab = np.ndarray(shape=[batch_size, self._max_enc_tok], dtype=np.int32)

        # Shuffle files at the start of an epoch.
        if self._ptr == 0:
            shuffle(files)

        # Start and end index for a batch of data.
        start = self._ptr
        end = self._ptr + batch_size
        self._ptr = end

        for i in range(start, end):
            j = i - start   # Index of the example in current batch.
            article_sents, summary = get_article_summary(files[i], art_format="sentences")

            # When chunking reshaping the data.
            if chunk:
                art_words = ' '.join(article_sents)

                article_lines = [[]]
                word_count = 0
                for word in art_words.split(' '):
                    article_lines[-1].append(word)
                    word_count += 1

                    if word_count == self._max_enc_tok_per_sent:
                        word_count = 0
                        article_lines.append([])

                article_sents = []
                for line in article_lines:
                    article_sents.append(' '.join(line))

            if len(article_sents) >= self._max_enc_sent:                # Truncate no. of sentences.
                if permutate:
                    indcs = sorted(sample(range(len(article_sents)), self._max_enc_sent))
                    article_sents = [article_sents[i] for i in indcs]
                else:
                    article_sents = article_sents[: self._max_enc_sent]
            else:
                article_sents += (self._max_enc_sent -
                                  len(article_sents)) * ['']            # Add empty sentences.

            enc_inp_ids = []
            enc_inp_tokens = []
            for sent_idx, art_sent in enumerate(article_sents):

                enc_sent_tokens = art_sent.split(' ')
                if len(enc_sent_tokens) >= self._max_enc_tok_per_sent:                  # Truncate no. of tokens.
                    if permutate:
                        indcs = sorted(sample(range(len(enc_sent_tokens)), self._max_enc_tok_per_sent))
                        enc_sent_tokens = [enc_sent_tokens[i] for i in indcs]
                    else:
                        enc_sent_tokens = enc_sent_tokens[: self._max_enc_tok_per_sent]
                else:
                    enc_sent_tokens += (self._max_enc_tok_per_sent
                                        - len(enc_sent_tokens)) * [params.PAD_TOKEN]    # Pad.

                # Encoder sentence representation in the fixed vocabulary.
                enc_sent_ids = [self._vocab.word2id(w) for w in enc_sent_tokens]

                # Appending to the lists.
                enc_inp_ids.append(enc_sent_ids)
                enc_inp_tokens += enc_sent_tokens

            # Summary tokens.
            sum_tokens = summary.split(' ')                                 # Summary tokens.
            sum_tokens = [t for t in sum_tokens if t not in
                          [params.SENTENCE_START, params.SENTENCE_END]]     # Removing <s>, </s> tokens.

            if len(sum_tokens) > self._max_dec_tok - 1:                     # Truncate.
                sum_tokens = sum_tokens[: self._max_dec_tok - 1]

            # Decoder Input.
            dec_inp_tokens = [params.START_DECODING] + sum_tokens
            if len(dec_inp_tokens) < self._max_dec_tok:
                dec_inp_tokens += (self._max_dec_tok - len(dec_inp_tokens)) * [params.PAD_TOKEN]

            # Decoder Output.
            dec_out_tokens = sum_tokens + [params.STOP_DECODING]
            if len(dec_out_tokens) < self._max_dec_tok:
                dec_out_tokens += (self._max_dec_tok - len(dec_out_tokens)) * [params.PAD_TOKEN]

            dec_inp_ids = [self._vocab.word2id(w) for w in dec_inp_tokens]
            dec_out_ids = [self._vocab.word2id(w) for w in dec_out_tokens]

            # Encoder input, decoder output representation in extended vocabulary.
            enc_inp_ids_ext_vocab = None
            if self._use_pgen:
                enc_inp_ids_ext_vocab, article_oovs = article2ids(enc_inp_tokens, self._vocab)
                dec_out_ids = summary2ids(dec_out_tokens, self._vocab, article_oovs)

                if len(article_oovs) > max_oov_size:
                    max_oov_size = len(article_oovs)

            # Appending to the batch of inputs.
            enc_inp[j] = np.array(enc_inp_ids).astype(np.int32)
            dec_inp[j] = np.array(dec_inp_ids).astype(np.int32)
            dec_out[j] = np.array(dec_out_ids).astype(np.int32)

            if self._use_pgen:
                enc_inp_ext_vocab[j] = np.array(enc_inp_ids_ext_vocab).astype(np.int32)

        # Resetting the pointer after the last batch.
        if self._ptr == num_examples:
            self._ptr = 0

        # Setting the pointer for the last batch.
        if self._ptr + batch_size > num_examples:
            self._ptr = num_examples - batch_size

        # Padding masks.
        pad_id = self._vocab.word2id(params.PAD_TOKEN)
        enc_pad_mask = (enc_inp != pad_id).astype(np.float32)               # B x S_in x T_in.
        enc_doc_mask = np.sum(enc_pad_mask, axis=2)                         # B x S_in.
        enc_doc_mask = np.greater(enc_doc_mask, 0).astype(np.float32)       # B x S_in.
        dec_pad_mask = (dec_out != pad_id).astype(np.float32)               # B x T_dec.

        batches = [enc_inp, enc_pad_mask, enc_doc_mask, dec_inp, dec_out, dec_pad_mask]
        if self._use_pgen:
            batches += [enc_inp_ext_vocab, max_oov_size]

        return batches

    def _get_test_batch(self, batch_size, chunk=False):
        num_examples = self.num_test_examples
        files = self._test_files

        enc_inp = np.ndarray(shape=[batch_size, self._max_enc_sent, self._max_enc_tok_per_sent], dtype=np.int32)

        max_oov_size = -np.infty
        enc_inp_ext_vocab = None
        if self._use_pgen:
            enc_inp_ext_vocab = np.ndarray(shape=[batch_size, self._max_enc_tok], dtype=np.int32)

        summaries = []              # Used in 'test' mode.
        ext_vocabs = []             # Extended vocabularies.

        # # Shuffle files at the start of an epoch.
        # if self._ptr == 0:
        #     shuffle(files)

        # Start and end index for a batch of data.
        start = self._ptr
        end = self._ptr + batch_size
        self._ptr = end

        for i in range(start, end):
            j = i - start
            article_sents, summary = get_article_summary(files[i], art_format="sentences", sum_format="sentences")

            # When chunking reshaping the data.
            if chunk:
                art_words = ' '.join(article_sents)

                article_lines = [[]]
                word_count = 0
                for word in art_words.split(' '):
                    article_lines[-1].append(word)
                    word_count += 1

                    if word_count == self._max_enc_tok_per_sent:
                        word_count = 0
                        article_lines.append([])

                article_sents = []
                for line in article_lines:
                    article_sents.append(' '.join(line))

            if len(article_sents) >= self._max_enc_sent:                # Truncate no. of sentences.
                article_sents = article_sents[: self._max_enc_sent]
            else:                                                       # Add empty sentences.
                article_sents += (self._max_enc_sent -
                                  len(article_sents)) * ['']

            enc_inp_ids = []
            enc_inp_tokens = []
            for sent_idx, art_sent in enumerate(article_sents):
                # Break if max. no. of encoder sentences is reached.
                if sent_idx >= self._max_enc_sent:
                    break

                enc_sent_tokens = art_sent.split(' ')
                if len(enc_sent_tokens) >= self._max_enc_tok_per_sent:                  # Truncate no. of tokens.
                    enc_sent_tokens = enc_sent_tokens[: self._max_enc_tok_per_sent]
                else:                                                                   # Pad.
                    enc_sent_tokens += (self._max_enc_tok_per_sent -
                                        len(enc_sent_tokens)) * [params.PAD_TOKEN]

                # Encoder representation in the fixed vocabulary.
                enc_sent_ids = [self._vocab.word2id(w) for w in enc_sent_tokens]

                # Appending to the lists.
                enc_inp_ids.append(enc_sent_ids)
                enc_inp_tokens += enc_sent_tokens

            # Encoder input representation in the extended vocabulary.
            enc_inp_ids_ext_vocab = None
            article_oovs = None
            if self._use_pgen:
                enc_inp_ids_ext_vocab, article_oovs = article2ids(enc_inp_tokens, self._vocab)

                if len(article_oovs) > max_oov_size:
                    max_oov_size = len(article_oovs)

            # Appending to the input batch.
            enc_inp[j] = np.array(enc_inp_ids).astype(np.int32)

            if self._use_pgen:
                enc_inp_ext_vocab[j] = np.array(enc_inp_ids_ext_vocab).astype(np.int32)
                ext_vocabs.append(article_oovs)

            # Summaries.
            summaries.append(summary)

        # Resetting the pointer after the last batch.
        if self._ptr == num_examples:
            self._ptr = 0

        # Setting the pointer for the last batch.
        if self._ptr + batch_size > num_examples:
            self._ptr = num_examples - batch_size

        # Padding masks.
        pad_id = self._vocab.word2id(params.PAD_TOKEN)
        enc_pad_mask = (enc_inp != pad_id).astype(np.float32)           # B x S_in x T_in.
        enc_doc_mask = np.sum(enc_pad_mask, axis=2)                     # B x S_in.
        enc_doc_mask = np.greater(enc_doc_mask, 0).astype(np.float32)   # B x S_in.

        indices = list(range(start, end))
        batches = [indices, summaries, files[start: end], enc_inp, enc_pad_mask, enc_doc_mask]
        if self._use_pgen:
            batches += [enc_inp_ext_vocab, ext_vocabs, max_oov_size]

        return batches


def average_gradients(tower_grads):
    """
        This function collects gradients from all towers and returns the average gradient.
    :param tower_grads: List of gradients from all towers.
    :return: List of average gradients of all variables.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # For M variables and N GPUs, tower_grads is of the form
        # ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1), ..., (grad0_gpuN, var0_gpuN))
        # ((grad1_gpu0, var1_gpu0), (grad1_gpu1, var1_gpu1), ..., (grad1_gpuN, var1_gpuN))
        # ......
        # ((gradM_gpu0, varM_gpu0), (gradM_gpu1, varM_gpu1), ..., (gradM_gpuN, varM_gpuN))

        grads = []
        for g, _ in grad_and_vars:
            # Adding an extra dimension for concatenation later.
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)  # Concatenation along added dimension.
        grad = tf.reduce_mean(grad, axis=0)  # Average gradient.

        # Variable name is same in all the GPUs. So, it suffices to use the one in the 1st GPU.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def eval_model(path_to_results):
    """
        This function is for the ROUGE evaluation.
    :return:
    """
    r = pyrouge.Rouge155()
    r.system_dir = path_to_results + 'predictions/'
    r.model_dir = path_to_results + 'groundtruths/'
    r.system_filename_pattern = '(\d+)_pred.txt'
    r.model_filename_pattern = '#ID#_gt.txt'
    rouge_results = r.convert_and_evaluate()

    rouge_dict = r.output_to_dict(rouge_results)
    rouge_log(rouge_dict, path_to_results)


def rouge_log(results_dict, path_to_results):
    """
        This function saves the rouge results into a file.
    :param results_dict: Dictionary output from pyrouge consisting of ROUGE results.
    :param path_to_results: Path where the results file has to be stored.
    :return:
    """
    log_str = ""
    for x in ["1", "2", "l"]:

        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)

    results_file = path_to_results + 'ROUGE_results.txt'
    with open(results_file, "w") as f:
        f.write(log_str)


def get_running_avg_loss(loss, running_avg_loss, decay=0.99):
    """
        This function updates the running averages loss.
    :param loss: Loss at the current step.
    :param running_avg_loss: Running average loss after the previous step.
    :param decay: The decay rate.
    :return:
    """

    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

    return running_avg_loss


def get_dependencies(tensor):
    dependencies = set()
    dependencies.update(tensor.op.inputs)
    for sub_op in tensor.op.inputs:
        dependencies.update(get_dependencies(sub_op))

    return dependencies


def get_placeholder_dependencies(tensor):
    dependencies = get_dependencies(tensor)
    dependencies = [tensor for tensor in dependencies if tensor.op.type == "Placeholder"]

    return dependencies
