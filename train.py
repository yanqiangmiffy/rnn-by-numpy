import subprocess
import preprocess
from rnn import RNN
import numpy as np
import pickle

dat = []
data = preprocess.dataset()
rnn = RNN(data.vocabulary_size)
x = data.X_train[20006]
print(x)
np.random.seed(10)
# Train on a small subset of the data to see what happens
print("正在训练...")
losses = RNN.train_with_sgd(rnn, data.X_train[:20000], data.Y_train[:20000], nepoch=10, evaluate_loss_after=1)

predict = rnn.predict(x)

print("predict shape = " + str(predict.shape))
print (predict)
array_of_words = " ".join([data.index_to_word[x] for x in predict ])

print(array_of_words)


with open('model.pkl','wb') as out:
    pickle.dump(rnn, out, pickle.HIGHEST_PROTOCOL)
