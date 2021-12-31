import pickle
import logging
import numpy as np
from pathlib import Path
from module import Embedding


def get_examples(file_name, embedding, vocab):
    mode = file_name.split('/')[-1][:-7]  # train validation test

    if isinstance(embedding, Embedding):
        cache_name = './data/examples/' + mode + '.glove.pickle'
    else:
        cache_name = './data/examples/' + mode + '.bert.pickle'

    if Path(cache_name).exists():
        file = open(cache_name, 'rb')
        examples = pickle.load(file)
        logging.info('Data from cache file: %s, total %d docs.' % (cache_name, len(examples)))
        return examples

    examples = []

    file = open(file_name, 'rb')
    data = pickle.load(file)

    for doc_data in data:
        # label
        label = doc_data[0]
        id = vocab.label2id(label)

        # words
        sents_data = doc_data[2]

        if isinstance(embedding, Embedding):
            doc = []
            for sent_data in sents_data:
                sent_words, graph = sent_data[0], sent_data[1]
                word_ids = vocab.word2id(sent_words)
                extword_ids = vocab.extword2id(sent_words)

                edges_index, edges_type = graph
                edges_type_id = vocab.type2id(edges_type)
                graph = (edges_index, edges_type_id)

                doc.append([graph, word_ids, extword_ids])
            examples.append([id, len(doc), doc])

        else:
            doc = []
            for sent_data in sents_data:
                sent_words, graph = sent_data[0], sent_data[1]
                token_ids, lens = embedding.tokenize(sent_words)

                edges_index, edges_type = graph
                edges_type_id = vocab.type2id(edges_type)
                graph = (edges_index, edges_type_id)

                doc.append([graph, lens, token_ids])
            examples.append([id, len(doc), doc])

    logging.info('Data from file: %s, total %d docs.' % (file_name, len(examples)))

    file = open(cache_name, 'wb')
    pickle.dump(examples, file)
    logging.info('Cache Data to file: %s, total %d docs.' % (cache_name, len(examples)))
    return examples


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, config, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        np.random.shuffle(data)

    lengths = [example[1] for example in data]
    noisy_lengths = [-(len_ + np.random.uniform(-noise, noise)) for len_ in lengths]
    sorted_indices = np.argsort(noisy_lengths).tolist()
    config.sorted_indices = sorted_indices
    sorted_data = [data[i] for i in sorted_indices]

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch
