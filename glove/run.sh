xa5%glove/build/vocab_count -min-count 2 -verbose 2 < ../data/questions_clean2_l.txt > vocab_clean2.txt
glove/build/cooccur -memory 20.0 -vocab-file vocab_clean2.txt -verbose 2  -window-size 15 < ../data/questions_clean2_l.txt > cooccurrence_clean2.bin
glove/build/shuffle -memory 20.0 -verbose 2  < cooccurrence_clean2.bin > cooccurrence_clean2.shuf.bin
glove/build/glove -save-file vectores_clean2 -threads 16 -input-file cooccurrence_clean2.shuf.bin -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file vocab_clean2.txt -verbose 2
