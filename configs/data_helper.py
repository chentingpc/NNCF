import numpy as np
import cPickle as pickle
from collections import defaultdict
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from numba import jit


class DataHelper(object):
    '''
    Major functions:
        1. load_data() for loading preprocessed data, then
            1.1 call get_Xy_seq() to get a train test split in sequence format
            1.2 call get_Xy_cmat() to get a train test split in count matrix format
        2. preprocess() for preprocessing raw text instances, and generate revs and vocabulary etc.

    utility functions:
        1. translate_word2id() / translate_id2word()
        2. transform_to_onehot()

    Some important data structures:
        preprocessed instances are stored as a list, named revs, each instance is like:
            datum = {"y": ,             # label
                     "text": ,          # original text
                     "text_mapped": ,   # mapped text using dictionry defined on vocabulary
                     "num_words": ,
                     "split": }         # used for (1) CV, (2) identifing train/val/test

    '''
    def __init__(self):
        pass

    def load_data(self, data_file, silence=True):
        '''
        Load preprocessed data_file.
        '''

        if not silence:
            print "loading data...",
        x = pickle.load(open(data_file, "rb"))
        revs, W_w2v, W_rand, word2id, id2word, vocab, label2id, info = [x[i] for i in range(8)]

        self.revs = revs
        self.W_w2v = W_w2v
        self.W_rand = W_rand
        self.word2id = word2id
        self.id2word = id2word
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = int(re.findall('max instance length: (\d+)', info)[0])
        try:
            self.cv = int(re.findall('cv = (\d+)', info)[0])
            self.cv_predefined = None
        except:
            self.cv = None
            self.cv_predefined = re.findall('data split is pregiven: ([a-zA-Z0-9,]+),',
                                            info)[0].strip().split(',')
        self.dim = int(re.findall('dim = (\d+)', info)[0])

        if not silence:
            print "data loaded!"
            print info

    def get_Xy_seq(self, split, max_len, max_filter_size=7, pad_at_beginning=False):
        '''
        Get the split of Xy as sequence
        --------------------------------------------

        split: int or string
            when cv is used, it is int indicating which split of cv to use as train/test
            otherwise the string will specify the main split as reference (usually 'train')

        max_len: int
            max truncated length of sentence

        max_filter_size: int
            the max filter size of one dimensional filter

        Return -

        when split is int (cv is used), return both train and test
        Otherwise return all predefined split
        '''

        def filter_unseen_word(X_reference, Xs):
            words = list(set(X_reference.flatten()))

            @jit
            def filter_unseen_word_at_once(words, X):
                m, n = X.shape
                for i in range(m):
                    for j in range(n):
                        if X[i, j] in words:
                            pass
                        else:
                            X[i, j] = 0
            for X in Xs:
                filter_unseen_word_at_once(words, X)

        if isinstance(split, int):
            assert split >= 0 and split <= self.cv, [split]
            self.X_train, self.y_train, self.X_test, self.y_test = \
                self._make_idx_data_cv(self.revs, split, max_len, max_filter_size, pad_at_beginning)
            filter_unseen_word(self.X_train, [self.X_test])
            return self.X_train, self.y_train, self.X_test, self.y_test
        else:
            assert self.cv is None
            split_predefined = self.cv_predefined
            reference_split = split
            assert reference_split in split_predefined
            X, y = self._make_idx_data(self.revs, max_len, max_filter_size, pad_at_beginning)
            X_other_splits = []
            for split in split_predefined:
                if split != reference_split:
                    X_other_splits.append(X[split])
            filter_unseen_word(X[reference_split], X_other_splits)
            self.X, self.y = X, y
            return X, y

    def _pad_and_truncate(self, instance, max_len, max_filter_size=5, pad_at_beginning=False):
        """
        Pad instance with zeroes, truncate it if the instance excceds max_len.
        """
        x = []
        pad = max_filter_size - 1
        if pad_at_beginning:
            instance_len = len(instance)
            if instance_len > max_len:
                instance_len = max_len
            for i in xrange(max_len + 2 * pad - instance_len):
                x.append(0)
        else:
            for i in xrange(pad):
                x.append(0)
        if len(instance) > max_len:
            x += instance[:max_len]
        else:
            x += instance
        if pad_at_beginning:
            assert len(x) == max_len + 2 * pad
        while len(x) < max_len + 2 * pad:
            x.append(0)
        return x

    def _make_idx_data_cv(self, revs, split, max_len, max_filter_size=5, pad_at_beginning=False):
        """
        Transforms instances(sentences/documents) into a 2-d matrix.
        """
        X_train, X_test = [], []
        y_train, y_test = [], []
        for rev in revs:
            instance = self._pad_and_truncate(rev["text_mapped"], max_len,
                                              max_filter_size, pad_at_beginning)
            if rev["split"] == split:
                X_test.append(instance)
                y_test.append(rev['y'])
            else:
                X_train.append(instance)
                y_train.append(rev['y'])
        X_train = np.array(X_train, dtype="int")
        X_test = np.array(X_test, dtype="int")
        y_train = np.array(y_train, dtype="int")
        y_test = np.array(y_test, dtype="int")
        return X_train, y_train, X_test, y_test

    def _make_idx_data(self, revs, max_len, max_filter_size=5, pad_at_beginning=False):
        """
        Transforms instances(sentences/documents) into a 2-d matrix., return all splits
        """
        X, y = {}, {}
        for rev in revs:
            split = rev['split']
            if split not in X:
                X[split] = []
                y[split] = []
        for rev in revs:
            split = rev['split']
            instance = self._pad_and_truncate(rev["text_mapped"], max_len,
                                              max_filter_size, pad_at_beginning)
            X[split].append(instance)
            y[split].append(rev['y'])
        for k, v in X.iteritems():
            X[k] = np.array(v, dtype="int")
        for k, v in y.iteritems():
            y[k] = np.array(v, dtype="int")
        return X, y

    def get_Xy_cmat(self, split, tf_idf=True, max_features=None, stop_words='english'):
        '''
        Get the split of the doc-word count matrix, where count can be tf_idf
        --------------------------------------------

        split: int or string
            when cv is used, it is int indicating which split of cv to use as train/test
            otherwise the string will specify the main split as reference (usually 'train')

        Return -

        when split is int (cv is used), return both train and test
        Otherwise return all predefined split
        '''
        revs = self.revs
        if isinstance(split, int):
            assert split >= 0 and split <= self.cv, [split]
            text_train, y_train, text_test, y_test = [], [], [], []
            for rev in revs:
                if rev['split'] == split:
                    text_test.append(rev['text'])
                    y_test.append(rev['y'])
                else:
                    text_train.append(rev['text'])
                    y_train.append(rev['y'])
            if tf_idf:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
            else:
                vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
            X_train = vectorizer.fit_transform(text_train)
            X_test = vectorizer.transform(text_test)
            self.X_train_cmat, self.y_train_cmat, self.X_test_cmat, self.y_test_cmat = \
                X_train, y_train, X_test, y_test
            return X_train, y_train, X_test, y_test
        else:
            assert self.cv is None
            split_predefined = self.cv_predefined
            reference_split = split
            assert reference_split in split_predefined
            text, X, y = {}, {}, {}
            for rev in revs:
                split = rev['split']
                if split not in text:
                    text[split] = []
                    X[split] = []
                    y[split] = []
            for rev in revs:
                split = rev['split']
                text[split].append(rev['text'])
                y[split].append(rev['y'])
            if tf_idf:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
            else:
                vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
            X[reference_split] = vectorizer.fit_transform(text[reference_split])
            for split in split_predefined:
                if split == reference_split:
                    continue
                X[split] = vectorizer.transform(text[split])
            self.X_cmat, self.y_cmat = X, y
            return X, y

    def preprocess(self, instances_json, cv, w2v_file, output, min_df=1, clean_string=True):
        '''
        preprocess raw text, label, split, and load word embeddings

        instances_json: list of dict
            require keys: label, text, split (required when cv is set None)

        cv: int or None
            if None, the split should be specify in instances_json
            otherwise, random split for cv

        w2v_file: string
            word2vec embedding file path

        output: string
            output filename

        min_df: int
            words to include: (1) appear in word2vec, (2) appear at least min_df in data instances
        '''
        print "transforming data...",
        revs, vocab, label2id = self._transform_text(instances_json, cv, clean_string)
        max_len = np.max(pd.DataFrame(revs)["num_words"])
        print "data transformed!"
        print "number of instances: " + str(len(revs))
        print "number of labels: " + str(len(label2id))
        print "vocab size: " + str(len(vocab))
        print "max instance length: " + str(max_len)
        print "loading word2vec vectors...",
        w2v = self._load_bin_vec(w2v_file, vocab)
        print "word2vec loaded!"
        print "num words already in word2vec: " + str(len(w2v))
        self._add_unknown_words(w2v, vocab, min_df)
        W_w2v, word2id, id2word = self._map_embedding_matrix(w2v)
        rand_vecs = {}
        self._add_unknown_words(rand_vecs, vocab, min_df)
        W_rand, _, _ = self._map_embedding_matrix(rand_vecs)
        self._map_text(revs, word2id)
        info = "number of instances: " + str(len(revs)) + "    \n"
        info += "number of labels: " + str(len(label2id)) + "    \n"
        info += "vocab size: " + str(len(vocab)) + "    \n"
        info += "max instance length: " + str(max_len) + "    \n"
        if cv is not None:
            info += "data split is cv = %d    \n" % cv
        else:
            info += 'data split is pregiven: '
            split_dict = dict()
            for rev in revs:
                split = rev['split']
                try:
                    split_dict[split] += 1
                except:
                    split_dict[split] = 1
            for split in split_dict:
                info += '%s,' % split
            info += '\n'
            for split, split_count in split_dict.iteritems():
                info += '%s\t%d\n' % (split, split_count)
        info += "embedding dim = %d" % 300
        with open(output, "wb") as fp:
            pickle.dump([revs, W_w2v, W_rand, word2id, id2word, vocab, label2id, info], fp,
                        pickle.HIGHEST_PROTOCOL)
        print "dataset created!"
        print info

    def _transform_text(self, instances_json, cv=None, clean_string=True):
        """
        Transform text by mapping vocabulary, label, and mark cv split.

        instances_json: list of dict
            require keys: label, text, split (required when cv is set None)

        cv: int or None
            if None, the split should be specify in instances_json
            otherwise, random split for cv

        Return:

        revs: data instances with json representation

        vocab: word => occured document count

        label2id: label => lable id
        """
        revs = []
        vocab = defaultdict(float)
        label2id = dict()
        max_lid = -1
        for instance_dict in instances_json:
            label = instance_dict['label']
            text = instance_dict['text']
            if cv is None:
                split = instance_dict['split']
            else:
                split = np.random.randint(0, cv)
            try:
                lid = label2id[label]
            except:
                max_lid += 1
                lid = max_lid
                label2id[label] = lid
            rev = [text]
            if clean_string:
                orig_text = self._clean_str(" ".join(rev))
            else:
                orig_text = " ".join(rev).lower()
            words_in_instance = set(orig_text.split())
            for word in words_in_instance:
                vocab[word] += 1
            datum = {"y": lid,
                     "text": orig_text,
                     "text_mapped": '',  # fullilled later
                     "num_words": len(orig_text.split()),
                     "split": split}
            revs.append(datum)
        return revs, vocab, label2id

    def _load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        --------------------------------------------------------

        fname: string
            path of the embedding file

        vocab: dict
            vocabulary of word => occured dobument count

        Return

        word_vecs: dict
            word => vector for word in vocab
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        return word_vecs

    def _map_embedding_matrix(self, word_vecs, k=300):
        """
        Get word embedding matrix. W[i] is the vector for word indexed by i

        Also, word2id, id2word, where word id starts from 1
        """
        vocab_size = len(word_vecs)
        word2id = dict()
        id2word = dict()
        W = np.zeros(shape=(vocab_size+1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word2id[word] = i
            id2word[i] = word
            i += 1
        return W, word2id, id2word

    def translate_word2id(self, instance_of_words, word2id=None):
        x = []
        if word2id is None:
            word2id = self.word2id
        for word in instance_of_words:
            if word in word2id:
                x.append(word2id[word])
        return x

    def translate_id2word(self, instance_of_ids, id2word=None):
        x = []
        if id2word is None:
            id2word = self.id2word
        for id in instance_of_ids:
            if id in id2word:
                x.append(id2word[id])
        return x

    def _map_text(self, revs, word2id):
        """ mapp text of raw words into word ids """
        for rev in revs:
            words_seq = rev["text"].split()
            rev["text_mapped"] = self.translate_word2id(words_seq, word2id)

    def _add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that not in word_vecs and occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

    def _clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[^A-Za-z0-9(),!?;\.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if TREC else string.strip().lower()

    def _clean_str_sst(self, string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[^A-Za-z0-9(),!?;\.\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def transform_to_onehot(self, seq, num_class):
        '''
        Transform a sequence of symbols into one hot encodding of sequence (seq_len x num_class)
        '''
        if isinstance(seq, np.ndarray):
            seq_len = seq.shape[0]
            assert len(seq.shape) == 1, seq.shape
        elif isinstance(seq, list):
            seq_len = len(seq)
        else:
            assert False
        y = np.zeros((seq_len, num_class))
        y[range(seq_len), seq] = 1
        return y
