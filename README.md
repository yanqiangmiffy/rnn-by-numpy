# rnn-by-numpy
使用numpy实现rnn和语言模型

1. 文章：
* [Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* [一文弄懂神经网络中的反向传播法——BackPropagation](https://www.cnblogs.com/charlotte77/p/5629865.html)

2. 内容：
本仓库主要用numpy从头开始构建rnn结构，包括`前向传播算法` `反向传播算法` `学习率` `随机梯度下降`;
![rnn](https://github.com/yanqiangmiffy/rnn-by-numpy/blob/master/images/rnn.jpg)

3. 实例：给定一个x来预测y，虽然这个没有实际意义，所以在这里主要目的是为了阐释rnn的算法
```
x:
SENTENCE_START what are n't you understanding about this ? !
[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]
 
y:
what are n't you understanding about this ? ! SENTENCE_END
[51, 27, 16, 10, 856, 53, 25, 34, 69, 1]

```
