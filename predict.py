import pickle
import nltk
import preprocess
unknown_token="UNKNOWN_TOKEN"
python_string="worked"
tokenized_sentence=nltk.word_tokenize(python_string.lower())

data=preprocess.dataset()
tokenized_sentence = [w if w in data.word_to_index else unknown_token for w in tokenized_sentence] # 将句子使用词汇表中的单词表示
x=[data.word_to_index[word] for word in tokenized_sentence]
print(x)


with open('model.pkl', 'rb') as model_data:
    print("加载训练的模型...")

    rnn=pickle.load(model_data)
    predict = rnn.predict(x)

    print("predict shape = " + str(predict.shape))
    print(predict)
    array_of_words = " ".join([data.index_to_word[x] for x in predict])

    print(array_of_words)