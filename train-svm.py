import os
import time
import json
import pickle
import logging
import argparse
from pathlib import Path
from src.utils import get_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', default=6, type=int, help='fold for test')
args = argparser.parse_args()

target_names = ['PI', 'FPCU', 'CT', 'TPSC', 'URC', 'DS', 'DR', 'IDT', 'SA', 'PC', 'PCI']


# train data
dir = './save/svm/' + str(args.fold)
model_file = dir + '/svm.model'

file = open('./data/svm/train_' + str(args.fold) + '.pickle', 'rb')
train_data = pickle.load(file)
logging.info('| {} | features {}'.format('train data ' + str(args.fold), train_data['text'].shape))

# model
start_time = time.time()

if Path(model_file).exists():
    # load
    file = open(model_file, 'rb')
    model = pickle.load(file)
    logging.info('| {} | times {:.2f}s'.format('load model', time.time() - start_time))
else:
    if not os.path.exists(dir):
        os.makedirs(dir)
    # train
    model = SVC(C=1.0, kernel="linear")
    model.fit(train_data['text'], train_data['label'])

    logging.info('| {} | times {:.2f}s'.format('train model', time.time() - start_time))

    # res
    file = open(model_file, 'wb')
    pickle.dump(model, file)
    file.close()
    logging.info('res model.')

# predict train
start_time = time.time()

y_pred_train = model.predict(train_data['text'])
score, f1 = get_score(train_data['label'], y_pred_train)
logging.info('| {} | score {} | f1 {}'.format('train', score, f1))
report = classification_report(train_data['label'], y_pred_train, digits=4, target_names=target_names)
logging.info('\n' + report)

logging.info('| {} | times {:.2f}s'.format('train', time.time() - start_time))

# predict dev
file = open('./data/svm/dev_' + str(args.fold) + '.pickle', 'rb')
dev_data = pickle.load(file)
logging.info('| {} | features {}'.format('dev data ' + str(args.fold), dev_data['text'].shape))

start_time = time.time()

y_pred_dev = model.predict(dev_data['text'])
score, f1 = get_score(dev_data['label'], y_pred_dev)
logging.info('| {} | score {} | f1 {}'.format('dev', score, f1))
report = classification_report(dev_data['label'], y_pred_dev, digits=4, target_names=target_names)
logging.info('\n' + report)

logging.info('| {} | times {:.2f}s'.format('dev', time.time() - start_time))

# predict tset
file = open('./data/svm/test_' + str(args.fold) + '.pickle', 'rb')
test_data = pickle.load(file)
logging.info('| {} | features {}'.format('test data ' + str(args.fold), test_data['text'].shape))

start_time = time.time()

y_pred_test = model.predict(test_data['text'])
score, f1 = get_score(test_data['label'], y_pred_test)
logging.info('| {} | score {} | f1 {}'.format('test', score, f1))
report = classification_report(test_data['label'], y_pred_test, digits=4, target_names=target_names)
logging.info('\n' + report)

logging.info('| {} | times {:.2f}s'.format('test', time.time() - start_time))

# save result
save_file = './res/svm/test_' + str(args.fold) + '.json'
file = open(save_file, 'w')
dic = {
    'ture_label': test_data['label'],
    'pred_label': y_pred_test.tolist()
}
file.write(json.dumps(dic))
file.close()

logging.info('| Save result to {}'.format(save_file))
