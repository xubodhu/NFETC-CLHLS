#!/bin/sh

echo "Downloading corpus for OntoNotes & Wiki"
wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip
rm corpus.zip

echo "Get preprocessed BBN corpus here manually"
# https://drive.google.com/file/d/1opjfoA0I2mOjE11kM_TYsaeq-HHO1rqv/view

echo "Downloading word embeddings..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
