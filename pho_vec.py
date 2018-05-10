# -*- coding: utf-8 -*-
'''
次内容包括 train数据集的phoneme提取成doc函数 build_phoneme_doc_dir
和 训练doc2vec在不同维度下的模型函数 build_doc_weidu
执行过程包括 首先执行build_phoneme_doc_dir 输入为训练数据集所在地址 此数据集为最原始数据集并非query _candiadte配对后的数据集
执行此文件之后便可以直接运行计算函数 生成对应文件即可不需要每次执行
'''

'''
gensim 2.0.0
'''
from gensim.models.doc2vec import Doc2Vec,LabeledSentence,TaggedDocument
import os
import string
import glob
import numpy as np

delset = string.punctuation
def build_doc_weidu(model):
    documents=[]
    even = []
    odd = []
    # walk = os.walk(os.path.realpath("pho/"))
    files = glob.glob('pho/*')
    for i in range(len(files)):
        fname = 'pho/{}'.format(i)
        with open(fname, 'r') as fin:
            content = fin.read().replace(' ', '')
        line_list = content.split('\n')
        # ids = list(range(len(line_list)//2))
        odd_lines = '\n'.join(line_list[::2])
        odd_lines = list(odd_lines)
        even_lines = '\n'.join(line_list[1::2])
        even_lines = list(even_lines)
        content = list(content)
        odd_vec = model.infer_vector(odd_lines)
        even_vec = model.infer_vector(even_lines)
        vec = model.infer_vector(content)
        odd.append(odd_vec)
        even.append(even_vec)
        documents.append(vec)
    return np.stack(documents), np.stack(odd), np.stack(even)


if __name__ == '__main__':
    doc_model = Doc2Vec.load('doc_model_/doc_phoneme_model_size125windows150')
    vectors, odds, evens = build_doc_weidu(doc_model)
    np.save('data/pho0.npy', vectors)
    np.save('data/pho1.npy', odds)
    np.save('data/pho2.npy', evens)