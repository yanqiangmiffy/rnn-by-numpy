# ==================================================
# nltk.sent_tokenize(text) #对文本按照句子进行分割
# nltk.word_tokenize(sent) #对句子进行分词，当输入的是包含多个句子的文档时，返回列表，每个列表包含对应句子的分词结果。
# ==================================================

import csv
import numpy as np
import itertools
import nltk
import pickle
import os

unknown_token="UNKNOWN_TOKEN"
sentence_start_token="SENTENCE_START"
sentence_end_token="SENTENCE_END"


class dataset:

    # 判断文件是否为空
    def file_exists(self,path):
        return os.path.exists(path)

    # 数据预处理
    def preprocess_data(self):
        # 读取数据，然后添加 SENTENCE_START 和 SENTENCE_END 标志
        print("正在读取 data/reddit-comments-2015-08.csv ....")
        with open('data/reddit-comments-2015-08.csv','r',encoding='utf-8') as f:
            reader=csv.reader(f,skipinitialspace=True) # 忽略字段定界符后面的空白符

            # 将所有comments分成句子
            sentences=itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])

            # 给句子添加添加 SENTENCE_START 和 SENTENCE_END 标志
            sentences=["%s %s %s" % (sentence_start_token,x,sentence_end_token) for x in sentences]
        print("一共处理了%s句子" % len(sentences))

        # 对句子进行分词
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # 统计词频
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("一共有%s个不同的词" % len(word_freq.items()))

        # 获取出现最多的前vocabulary_size个单词，然后构建它们的单词和索引映射矩阵 index_to_word,word_to_index
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(word, i) for i, word in enumerate(self.index_to_word)])
        print("正在使用的词汇表大小为%d" % self.vocabulary_size)
        print("词汇表出现的最后一个单词为%s 并且它的出现次数为%d" % (vocab[-1][0], vocab[-1][1]))

        # 将所有没有出现在词汇表中的单词替换成unknown_token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.word_to_index else unknown_token for w in sent]

        print("处理前的句子：%s" % sentences[0])
        print("处理后的句子：%s" % tokenized_sentences[0])

        # 构建训练数据
        # x:SENTENCE_START  what are n't you understanding about this ? !
        # [SENTENCE_START, 51, 27, 16, 10, 856, 53, 25, 34, 69]
        # y:what are n't you understanding about this ? ! SENTENCE_END
        # [51, 27, 16, 10, 856, 53, 25, 34, 69, SENTENCE_END]

        self.X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        self.Y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    def __init__(self):
        self.vocabulary_size=8000

        # 判断是否训练过模型
        print("原先的训练数据：" +str(self.file_exists('train.pkl')))

        if((self.file_exists('train.pkl')) and input("是否加载已经存在的数据?\n1.Yes\n2.No\n"))=='1':
            with open('train.pkl','rb') as in_data:
                print("正在加载数据...")
                self.X_train=pickle.load(in_data)
                self.Y_train=pickle.load(in_data)
                self.vocabulary_size=pickle.load(in_data)
                self.index_to_word=pickle.load(in_data)
                self.word_to_index=pickle.load(in_data)
        else:
            print("正在生成新数据")
            with open('train.pkl','wb') as out_data:
                self.preprocess_data()
                print("正在保存数据...")
                pickle.dump(self.X_train,out_data,pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.Y_train,out_data,pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vocabulary_size,out_data,pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.index_to_word,out_data,pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.word_to_index,out_data,pickle.HIGHEST_PROTOCOL)
                out_data.flush()
if __name__ == '__main__':
   data=dataset()
   print(data.word_to_index)