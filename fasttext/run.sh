#~/fastText/fasttext skipgram -input ../data/questions_clean_l.txt -output model_tmp -thread 16 -epoch 100
#~/fastText/fasttext print-vectors model_clean.bin < ../data/questions_clean.txt > vectors_clean.txt


~/fastText/fasttext supervised -input "../data/train_fast_trn2.csv" -output "result"  -thread 16 -epoch 100
