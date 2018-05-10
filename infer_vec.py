# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec,LabeledSentence,TaggedDocument
import os
import string
import glob
import numpy as np

delset = string.punctuation
def build_doc_weidu(model):
    documents=[]
    # walk = os.walk(os.path.realpath("doc/"))
    files = glob.glob('doc/*')
    for i in range(len(files)):
        fname = 'doc/{}'.format(i)
        with open(fname, 'r') as fin:
            content = fin.read().replace('  ', ' ')
        content = content.split()
        vec = model.infer_vector(content)
        documents.append(vec)

    return np.stack(documents)

if __name__ == '__main__':
    doc_model = Doc2Vec.load('doc_model_/doc_model_size125')
    vectors = build_doc_weidu(doc_model)
    np.save('data/doc.npy', vectors)


