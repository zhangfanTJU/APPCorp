# SVM
python train-svm.py --fold 9

# GloVe
## HAN
python train.py --config_file glove.cfg --emb glove --gpu 0 --fold 9

## HGAT
python train.py --config_file glove.cfg --emb glove --use_graph --gpu 0 --fold 9

# BERT
## HAN
python train.py --config_file bert.cfg --emb bert --gpu 0 --fold 9

## HGAT
python train.py --config_file bert.cfg --emb bert --use_graph --gpu 0 --fold 9
