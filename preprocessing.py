import pickle
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from src.vocab import Vocab

import logging
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('./emb/stanford_corenlp')

from collections import Counter

label_counter = Counter()

def build_graph(s):
    trees = nlp.dependency_parse(s)
    edges_index = []
    edges_type = []
    src_idx = []
    tgt_idx = []
    for tree in trees:
        label, src, tgt = tree
        if src == 0:
            continue
        src_idx.append(src - 1)
        tgt_idx.append(tgt - 1)
        edges_type.append(label)
        label_counter[label] += 1

    edges_index = [src_idx, tgt_idx]
    return (edges_index, edges_type)


def convert_data2tfidf(files):
    label2id = {
        'policy_introductory': 0,
        'first_party_collection_and_use': 1,
        'cookies_and_similar_technologies': 2,
        'third_party_share_and_collection': 3,
        'user_right_and_control': 4,
        'data_security': 5,
        'data_retention': 6,
        'international_data_transfer': 7,
        'specific_audiences': 8,
        'policy_change': 9,
        'policy_contact_information': 10
    }

    def read_data(data):
        texts = []
        labels = []
        for doc in data:
            labels.append(label2id[doc[0]])
            text = []
            sents_data = doc[2]
            for sent_data in sents_data:
                text.extend(sent_data[0])
            texts.append(' '.join(text))
        return texts, labels

    for filename in files:
        file = open('./data/preprocess/' + filename + '.pickle', 'rb')
        data = pickle.load(file)
        texts, labels = read_data(data)

        if filename.startswith("train"):
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
            arrays = vectorizer.fit_transform(texts).toarray()
        else:
            arrays = vectorizer.transform(texts).toarray()

        dic = {'text': arrays, 'label': labels}
        file = open('./data/svm/' + filename + '.pickle', 'wb')
        pickle.dump(dic, file)
        file.close()

        print(filename, arrays.shape)


def convert_fold2data():
    def read_data(file_name):
        docs = []
        doc = []
        label_flag = True
        f = open(file_name, 'r', encoding='UTF-8')
        lines = f.readlines()
        lines = list(map(lambda x: x.strip().lower(), lines))
        for line in lines:
            if line == '':
                if len(doc) > 0:
                    docs.append([label, len(doc), doc])
                # if len(docs) > 2:
                #     break
                doc = []
                label_flag = True
            else:
                if label_flag:
                    label = line
                    label_flag = False
                else:
                    words = nlp.word_tokenize(line)
                    graph = build_graph(line)
                    doc.append([words, graph])

        if len(doc) > 0:
            docs.append([label, len(doc), doc])

        return docs

    def write_data(file_name, data):
        file = open(file_name, 'wb')
        pickle.dump(data, file)
        file.close()
        print(file_name, len(data))

    lens = []
    for fold in range(10):
        # test
        test_data = read_data('./data/privacypolicy.test_' + str(fold))
        file_name = './data/preprocess/test_' + str(fold) + '.pickle'
        write_data(file_name, test_data)

        # dev
        dev_data = read_data('./data/privacypolicy.dev_' + str(fold))
        file_name = './data/preprocess/dev_' + str(fold) + '.pickle'
        write_data(file_name, dev_data)

        # train
        train_data = read_data('./data/privacypolicy.train_' + str(fold))
        train_name = './data/preprocess/train_' + str(fold) + '.pickle'
        write_data(train_name, train_data)

        lens.append(str([fold, len(train_data), len(dev_data), len(test_data)])[1:-1])

    for fold in range(10):
        print(lens[fold])


if __name__ == "__main__":
    convert_fold2data()

    # convert_data
    for fold in range(10):
        cache_name = "./save/vocab/" + str(fold) + ".pickle"
        train = "train_" + str(fold)
        dev = "dev_" + str(fold)
        test = "test_" + str(fold)
        files = [train, dev, test]

        # biuld vocab
        if Path(cache_name).exists():
            vocab_file = open(cache_name, 'rb')
            vocab = pickle.load(vocab_file)
            print('Load vocab from ' + cache_name)
        else:
            vocab = Vocab('./data/' + train + '.pickle')
            file = open(cache_name, 'wb')
            pickle.dump(vocab, file)
            print('Save vocab to ' + cache_name)

        # data2tfidf
        convert_data2tfidf(files)
