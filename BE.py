from flask import Flask, request, jsonify
import os
import sys
 
import paddle
import paddle.nn.initializer as I
 
import paddle.nn as nn
import paddle.nn.functional as F
import csv
import glob
import io
import json
import queue
import random
import time
from random import shuffle
from threading import Thread
 
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors

train_data_path = "./finished_files/chunked/train_*.json"
eval_data_path = "./finished_files/val.json"
decode_data_path = "./finished_files/test.json"
vocab_path = r'C:\Users\bebed\OneDrive\Desktop\Data Science\model\vocab'
log_root = "./log"
 
# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 1
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 200000
 
lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
 
pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0
 
eps = 1e-12
max_iterations = 100000
 
lr_coverage = 0.15

random.seed(123)
 
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"
 
PAD_TOKEN = "[PAD]"  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = "[UNK]"  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = "[START]"  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = "[STOP]"  # This has a vocab id, which is used at the end of untruncated target sequences
 
# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.
 
 
class Example(object):
    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)
 
        # Process the article
        article_words = article.split()
        if len(article_words) > max_enc_steps:
            article_words = article_words[: max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [
            vocab.word2id(w) for w in article_words
        ]  # list of word ids; OOVs are represented by the id for UNK token
 
        # Process the abstract
        abstract = " ".join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [
            vocab.word2id(w) for w in abstract_words
        ]  # list of word ids; OOVs are represented by the id for UNK token
 
        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            abs_ids, max_dec_steps, start_decoding, stop_decoding
        )
        self.dec_len = len(self.dec_input)
 
        # If using pointer-generator mode, we need to store some extra info
        if pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)
 
            # Get a version of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = abstract2ids(abstract_words, vocab, self.article_oovs)
 
            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(
                abs_ids_extend_vocab, max_dec_steps, start_decoding, stop_decoding
            )
 
        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences
 
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target
 
    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)
 
    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)
 
 
class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings
 
    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
 
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
 
        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)
 
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
 
        # For pointer-generator mode, need to store some extra info
        if pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]
 
    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(max_dec_steps, self.pad_id)
 
        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
 
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1
 
    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of lists
 
 
class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold
 
    def __init__(self, text, vocab, mode, batch_size, single_pass):
        self.text = text
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)
 
        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = (
                1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            )
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = (
                1  # 100 # how many batches-worth of examples to load into cache before bucketing
            )
 
        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()
 
        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()
 
    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            print(
                "Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i"
                % (self._batch_queue.qsize(), self._example_queue.qsize())
            )
            if self._single_pass and self._finished_reading:
                print("Finished reading dataset in single_pass mode.")
                return None
 
        batch = self._batch_queue.get()  # get the next Batch
        return batch
 
    def fill_example_queue(self):
        # input_gen = self.text_generator(example_generator(self._data_path, self._single_pass))
 
        # while True:
            # try:
            #     (article, abstract) = next(
            #         input_gen
            #     )  # read the next example from file. article and abstract are both strings.
            # except StopIteration:  # if there are no more examples:
            #     print("The example generator for this example queue filling thread has exhausted data.")
            #     if self._single_pass:
            #         print("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
            #         self._finished_reading = True
            #         break
            #     else:
            #         raise Exception("single_pass mode is off but the example generator is out of data; error.")

        abstract_sentences = [
            sent.strip() for sent in abstract2sents("")
        ]  # Use the <s> and </s> tags in abstract to get a list of sentences.
        example = Example(self.text, abstract_sentences, self._vocab)  # Process into an Example.
        self._example_queue.put(example)  # place the Example in the example queue.
 
    def fill_batch_queue(self):
        while True:
            if self.mode == "decode":
                # Beam search decode mode where a single example is repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(
                    inputs, key=lambda inp: inp.enc_len, reverse=True
                )  # sort by length of encoder sequence
 
                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i : i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
 
    def watch_threads(self):
        while True:
            print(
                "Bucket queue size: %i, Input queue size: %i"
                % (self._batch_queue.qsize(), self._example_queue.qsize())
            )
 
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print("Found example queue thread dead. Restarting.")
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print("Found batch queue thread dead. Restarting.")
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
 
    def text_generator(self, example_generator):
        while True:
            e = next(example_generator)
            try:
                article_text = e["article"]
                abstract_text = e["abstract"]
            except ValueError:
                print("Failed to get article or abstract from example")
                continue
            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                print("Found an example with empty article text. Skipping it.")
                continue
            else:
                yield (article_text, abstract_text)
 
 
class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
 
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
 
        # Read the vocab file and add words up to max_size
        with open(vocab_file, "r", encoding="utf-8", errors="replace") as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print("Warning: incorrectly formatted line in vocabulary file: %s\n" % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        "<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn't be in the vocab file, but %s is" % w
                    )
                if w in self._word_to_id:
                    raise Exception("Duplicated word in vocabulary file: %s" % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print(
                        "max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                        % (max_size, self._count)
                    )
                    break
 
        print(
            "Finished constructing vocabulary of %i total words. Last word added: %s"
            % (self._count, self._id_to_word[self._count - 1])
        )
 
    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]
 
    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self._id_to_word[word_id]
 
    def size(self):
        return self._count
 
    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ["word"]
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})
 
 
def example_generator(data_path, single_pass):
    while True:
        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, "Error: Empty filelist at %s" % data_path  # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = io.open(f, "r", encoding="utf-8")
            while True:
                reader_str = reader.readline()
                if reader_str == "":
                    break
                reader_json = json.loads(reader_str)
                yield reader_json
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break
 
 
def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs
 
 
def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids
 
 
def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError:  # w is OOV
            # article_oov_idx = i - vocab.size()
            assert (
                article_oovs is not None
            ), "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError:  # i doesn't correspond to an article oov
                raise ValueError(
                    "Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs"
                    % (i, article_oov_idx, len(article_oovs))
                )
        words.append(w)
    return words
 
 
def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START) : end_p])
        except ValueError:  # no more sentences
            return sents
 
 
def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(" ")
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = " ".join(words)
    return out_str
 
 
def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(" ")
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = " ".join(new_words)
    return out_str


def paddle2D_scatter_add(x_tensor, index_tensor, update_tensor, dim=0):
    dim0, dim1 = update_tensor.shape
    update_tensor = paddle.flatten(update_tensor, start_axis=0, stop_axis=1)
    index_tensor = paddle.reshape(index_tensor, [-1, 1])
    if dim == 0:
        index_tensor = paddle.concat(x=[index_tensor, (paddle.arange(dim1 * dim0) % dim0).unsqueeze(1)], axis=1)
    elif dim == 1:
        index_tensor = paddle.concat(x=[(paddle.arange(dim1 * dim0) // dim1).unsqueeze(1), index_tensor], axis=1)
    output_tensor = paddle.scatter_nd_add(x_tensor, index_tensor, update_tensor)
    return output_tensor
 
 
class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
 
        # Initialized embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(std=trunc_norm_init_std)),
        )
 
        # Initialized lstm weights
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=1,
            direction="bidirect",
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-rand_unif_init_mag, high=rand_unif_init_mag)
            ),
            bias_ih_attr=paddle.ParamAttr(initializer=I.Constant(value=0.0)),
        )
 
        # Initialized linear weights
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias_attr=False)
 
    # The variable seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        self.embedded = embedded
 
        output, hidden = self.lstm(embedded, sequence_length=paddle.to_tensor(seq_lens, dtype="int32"))
 
        encoder_feature = paddle.reshape(output, [-1, 2 * hidden_dim])  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)
 
        return output, encoder_feature, hidden
 
 
class ReduceState(paddle.nn.Layer):
    def __init__(self):
        super(ReduceState, self).__init__()
 
        self.reduce_h = nn.Linear(
            hidden_dim * 2,
            hidden_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(std=trunc_norm_init_std)),
        )
        self.reduce_c = nn.Linear(
            hidden_dim * 2,
            hidden_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(std=trunc_norm_init_std)),
        )
 
    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = paddle.reshape(h.transpose([1, 0, 2]), [-1, hidden_dim * 2])
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = paddle.reshape(c.transpose([1, 0, 2]), [-1, hidden_dim * 2])
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
 
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim
 
 
class Attention(paddle.nn.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        # Attention
        if is_coverage:
            self.W_c = nn.Linear(1, hidden_dim * 2, bias_attr=False)
        self.decode_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias_attr=False)
 
    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = encoder_outputs.shape
 
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = paddle.expand(dec_fea.unsqueeze(1), [b, t_k, n])  # B x t_k x 2*hidden_dim
        dec_fea_expanded = paddle.reshape(dec_fea_expanded, [-1, n])  # B * t_k x 2*hidden_dim
 
        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if is_coverage:
            coverage_input = paddle.reshape(coverage, [-1, 1])  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature
 
        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = paddle.reshape(scores, [-1, t_k])  # B x t_k
 
        attn_dist_ = F.softmax(scores, axis=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        # attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist_ / (
            paddle.reshape(normalization_factor, [-1, 1])
            + paddle.ones_like(paddle.reshape(normalization_factor, [-1, 1])) * sys.float_info.epsilon
        )
        # See the issue: https://github.com/atulkum/pointer_summarizer/issues/54
 
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = paddle.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = paddle.reshape(c_t, [-1, hidden_dim * 2])  # B x 2*hidden_dim
 
        attn_dist = paddle.reshape(attn_dist, [-1, t_k])  # B x t_k
 
        if is_coverage:
            coverage = paddle.reshape(coverage, [-1, t_k])
            coverage = coverage + attn_dist
 
        return c_t, attn_dist, coverage
 
 
class Decoder(paddle.nn.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # Decoder
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(std=trunc_norm_init_std)),
        )
 
        self.x_context = nn.Linear(hidden_dim * 2 + emb_dim, emb_dim)
 
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=1,
            direction="forward",
            weight_ih_attr=paddle.ParamAttr(
                initializer=I.Uniform(low=-rand_unif_init_mag, high=rand_unif_init_mag)
            ),
            bias_ih_attr=paddle.ParamAttr(initializer=I.Constant(value=0.0)),
        )
 
        if pointer_gen:
            self.p_gen_linear = nn.Linear(hidden_dim * 4 + emb_dim, 1)
 
        self.out1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out2 = nn.Linear(
            hidden_dim,
            vocab_size,
            weight_attr=paddle.ParamAttr(initializer=I.Normal(std=trunc_norm_init_std)),
        )
 
    def forward(
        self,
        y_t_1,
        s_t_1,
        encoder_outputs,
        encoder_feature,
        enc_padding_mask,
        c_t_1,
        extra_zeros,
        enc_batch_extend_vocab,
        coverage,
        step,
    ):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = paddle.concat(
                (
                    paddle.reshape(h_decoder, [-1, hidden_dim]),
                    paddle.reshape(c_decoder, [-1, hidden_dim]),
                ),
                1,
            )  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(
                s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage
            )
            coverage = coverage_next
 
        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(paddle.concat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
 
        h_decoder, c_decoder = s_t
        s_t_hat = paddle.concat(
            (paddle.reshape(h_decoder, [-1, hidden_dim]), paddle.reshape(c_decoder, [-1, hidden_dim])), 1
        )  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(
            s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage
        )
 
        if self.training or step > 0:
            coverage = coverage_next
 
        p_gen = None
        if pointer_gen:
            p_gen_input = paddle.concat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)
 
        output = paddle.concat((paddle.reshape(lstm_out, [-1, hidden_dim]), c_t), 1)  # B x hidden_dim * 3
        output1 = self.out1(output)  # B x hidden_dim
        output2 = self.out2(output1)  # B x vocab_size
        vocab_dist = F.softmax(output2, axis=1)
 
        if pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
 
            if extra_zeros is not None:
                vocab_dist_ = paddle.concat([vocab_dist_, extra_zeros], 1)
            final_dist = paddle2D_scatter_add(vocab_dist_, enc_batch_extend_vocab, attn_dist_, 1)
        else:
            final_dist = vocab_dist
 
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
 
 
class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        super(Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()
 
        # Shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
 
        if paddle.distributed.get_world_size() > 1:
            encoder = paddle.DataParallel(encoder)
            decoder = paddle.DataParallel(decoder)
            reduce_state = paddle.DataParallel(reduce_state)
 
        if is_eval:
            encoder.eval()
            decoder.eval()
            reduce_state.eval()
 
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
 
        if model_file_path is not None:
            self.decoder.set_state_dict(paddle.load(os.path.join(model_file_path, "decoder.params")))
            self.encoder.set_state_dict(paddle.load(os.path.join(model_file_path, "encoder.params")))
            self.reduce_state.set_state_dict(paddle.load(os.path.join(model_file_path, "reduce_state.params")))
 
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reduce_state": self.reduce_state.state_dict(),
        }
 
        #print(state_dict)
 
        self.state_dict = state_dict
        paddle.save(state_dict, "your_model.bin")

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
 
    def extend(self, token, log_prob, state, context, coverage):
        return Beam(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            context=context,
            coverage=coverage,
        )
 
    @property
    def latest_token(self):
        return self.tokens[-1]
 
    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)
 
 
class BeamSearch(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(vocab_path, vocab_size)
        self.model = Model(model_file_path, is_eval=True)
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)
 
    def decode(self, text):
        counter = 0
        batcher = Batcher(
        text, self.vocab, mode="decode", batch_size=beam_size, single_pass=True
        )
        # ex = Example(text, [], beam_search_processor.vocab)  # Process into an Example.
        # b = [ex for _ in range(batch_size)]
        # batch = Batch(b, self.vocab, batch_size)
        # Run beam search to get best Hypothesis
        batch = batcher.next_batch()
        
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)
 
            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
 
            decoded_words = outputids2words(
                output_ids, self.vocab, (batch.art_oovs[0] if pointer_gen else None)
            )
 
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
 
            original_abstract_sents = batch.original_abstracts_sents[0]
            print(decoded_words)
            result_string = ' '.join(decoded_words)
            return result_string

 
    def beam_search(self, batch):
        # The batch should have only one example
        (
            enc_batch,
            enc_padding_mask,
            enc_lens,
            enc_batch_extend_vocab,
            extra_zeros,
            c_t_0,
            coverage_t_0,
        ) = get_input_from_batch(batch)
 
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)
 
        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()
 
        # Prepare decoder batch
        beams = [
            Beam(
                tokens=[self.vocab.word2id(START_DECODING)],
                log_probs=[0.0],
                state=(dec_h[0], dec_c[0]),
                context=c_t_0[0],
                coverage=(coverage_t_0[0] if is_coverage else None),
            )
            for _ in range(beam_size)
        ]
 
        print(beams)
        results = []
        steps = 0
        while steps < max_dec_steps and len(results) < beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [
                t if t < self.vocab.size() else self.vocab.word2id(UNKNOWN_TOKEN) for t in latest_tokens
            ]
            y_t_1 = paddle.to_tensor(latest_tokens)
            all_state_h = []
            all_state_c = []
 
            all_context = []
 
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
 
                all_context.append(h.context)
 
            s_t_1 = (paddle.stack(all_state_h, 0).unsqueeze(0), paddle.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = paddle.stack(all_context, 0)
 
            coverage_t_1 = None
            if is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = paddle.stack(all_coverage, 0)
 
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(
                y_t_1,
                s_t_1,
                encoder_outputs,
                encoder_feature,
                enc_padding_mask,
                c_t_1,
                extra_zeros,
                enc_batch_extend_vocab,
                coverage_t_1,
                steps,
            )
            log_probs = paddle.log(final_dist)
            topk_log_probs, topk_ids = paddle.topk(log_probs, beam_size * 2)
 
            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
 
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = coverage_t[i] if is_coverage else None
 
                for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(
                        token=int(topk_ids[i, j]),
                        log_prob=float(topk_log_probs[i, j]),
                        state=state_i,
                        context=context_i,
                        coverage=coverage_i,
                    )
                    all_beams.append(new_beam)
 
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(STOP_DECODING):
                    if steps >= min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == beam_size or len(results) == beam_size:
                    break
 
            steps += 1
 
        if len(results) == 0:
            results = beams
 
        beams_sorted = self.sort_beams(results)
 
        return beams_sorted[0]

def get_input_from_batch(batch):
    batch_size = len(batch.enc_lens)
    enc_batch = paddle.to_tensor(batch.enc_batch, dtype="int64")
    enc_padding_mask = paddle.to_tensor(batch.enc_padding_mask, dtype="float32")
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None
 
    if pointer_gen:
        enc_batch_extend_vocab = paddle.to_tensor(batch.enc_batch_extend_vocab, dtype="int64")
        # The variable max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = paddle.zeros((batch_size, batch.max_art_oovs))
 
    c_t_1 = paddle.zeros((batch_size, 2 * hidden_dim))
 
    coverage = None
    if is_coverage:
        coverage = paddle.zeros(enc_batch.shape)
 
    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
 
 
def get_output_from_batch(batch):
    dec_batch = paddle.to_tensor(batch.dec_batch, dtype="int64")
    dec_padding_mask = paddle.to_tensor(batch.dec_padding_mask, dtype="float32")
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = paddle.to_tensor(dec_lens, dtype="float32")
 
    target_batch = paddle.to_tensor(batch.target_batch, dtype="int64")
 
    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


model_directory = r'C:\Users\bebed\OneDrive\Desktop\Data Science\model'

beam_search_processor = BeamSearch(model_directory)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Your AI model prediction function
def predict(data):
    # Replace this with your actual AI model logic
    print("Prediction started")
    result = beam_search_processor.decode(data["text"])
    print("Prediction ended")
    return result

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json  # assuming JSON data is sent from Angular
    result = predict(data)
    response = jsonify(result)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

if __name__ == '__main__':
    app.run(debug=True)