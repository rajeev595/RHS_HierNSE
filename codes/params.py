# -*- coding: utf-8 -*-
dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'

# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

# Vocabulary id's for sentence start, end, pad, unknown token, start and stop decoding.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'
VOCAB_SIZE = 200000
