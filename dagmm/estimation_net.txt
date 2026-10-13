# -*- coding: utf-8 -*-
import tensorflow as tf

class EstimationNet:
    """ Estimation Network

    This network converts input feature vector to softmax probability.
    Bacause loss function for this network is not defined,
    it should be implemented outside of this class.
    """

    # 这个网络（Estimation Network）的作用是将输入的特征向量转换为 softmax 概率。
    # 这个类本身没有定义损失函数（loss function），因此损失函数的计算和模型的训练过程需要在类的外部实现。

    def __init__(self, hidden_layer_sizes, activation=tf.nn.relu):

        # self: 代表类自身的实例，是所有类方法的第一个参数。
        # hidden_layer_sizes: 一个参数，预期会是一个整数列表（list of int），用来定义网络中每个隐藏层的大小（即神经元的数量）。
        # activation=tf.nn.relu: 一个可选参数，用于指定隐藏层的激活函数。它有一个默认值 tf.nn.relu（修正线性单元），这是目前最常用的激活函数之一。

        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of sizes of hidden layers.
            For example, if the sizes are [n1, n2],
            layer sizes of the network are:
            input_size -> n1 -> n2
            (network outputs the softmax probabilities of "n2" layer)
        activation : function
            activation function of hidden layer.
            the funtcion of last layer is softmax function.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def inference(self, z, dropout_ratio=None):

        # 在机器学习中，“推理（inference）”通常指模型接收输入数据并产生输出（预测）的过程，即前向传播。
        # z: 输入数据，它是一个 TensorFlow 张量（Tensor），形状通常是 (样本数, 特征数)。
        # dropout_ratio=None: 一个可选参数，用于控制 Dropout 的比率。Dropout 是一种防止模型过拟合的技术。如果这个值为 None，则不使用 Dropout。


        """ Output softmax probabilities

        Parameters
        ----------
        z : tf.Tensor shape : (n_samples, n_features)
            Data inferenced by this network
        dropout_ratio : tf.Tensor shape : 0-dimension float (optional)
            Specify dropout ratio
            (if None, dropout is not applied)

        Results
        -------
        probs : tf.Tensor shape : (n_samples, n_classes)
            Calculated probabilities
        """

        # 使用 tf.variable_scope("EstNet")。这是一个 TensorFlow 1.x 的概念，用于创建变量的命名空间。
        # 在这个 with 块中创建的所有 TensorFlow 变量（如网络层的权重和偏置），其名称都会自动添加 "EstNet/" 前缀。这有助于在复杂的模型中组织变量，并实现变量共享。

        with tf.variable_scope("EstNet"):

            # n_layer = 0: 初始化一个层计数器。
            # for size in self.hidden_layer_sizes[:-1]: 开始一个循环，遍历 hidden_layer_sizes 列表除了最后一个元素之外的所有元素。
            # 最后一个元素被保留用作输出层的神经元数量。

            n_layer = 0
            for size in self.hidden_layer_sizes[:-1]:

                # 在循环内部构建每一个隐藏层。
                # n_layer += 1: 增加层计数。
                # tf.layers.dense(...): 这是 TensorFlow 1.x Layers API 中用于创建全连接层（也叫稠密层）的函数。
                # 第一个参数 z 是这一层的输入。
                # 第二个参数 size 是这一层的神经元数量。
                # activation=self.activation 指定使用之前保存的激活函数（默认为 ReLU）。
                # name=... 为这一层赋予一个唯一的名称，如 "layer_1", "layer_2" 等。
                # z = ...: 将当前层的输出重新赋值给变量 z，这个输出将作为下一个循环中下一层的输入。

                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))

                # 如果 dropout_ratio 被指定了（不是 None），则调用 tf.layers.dropout()。
                # 这个函数会以 dropout_ratio 的概率随机地将该层的一些输出神经元置为零，这是一种有效的正则化手段，通常只在训练阶段使用。
                if dropout_ratio is not None:
                    z = tf.layers.dropout(z, dropout_ratio,
                        name="drop_{}".format(n_layer))

            # Last layer uses linear function (=logits)

            # 构建输出层（在循环结束后）。
            # size = self.hidden_layer_sizes[-1]: 获取 hidden_layer_sizes 列表中的最后一个值，作为输出层的神经元数量（通常等于分类任务的类别数）。
            # logits = tf.layers.dense(...): 再次使用 dense 函数创建输出层。
            # 这里的 activation=None。这意味着输出层是线性的，不应用任何激活函数。这个未经激活的原始输出通常被称为 logits。
            # 这是计算交叉熵损失（cross-entropy loss）前的标准做法。
            size = self.hidden_layer_sizes[-1]
            logits = tf.layers.dense(z, size, activation=None, name="logits")

            # Softmax output

            # 将 logits 转换为概率。
            # tf.contrib.layers.softmax(logits): tf.contrib 是 TensorFlow 1.x 中包含实验性或即将被废弃功能的模块。
            # 这里的 softmax 函数接收 logits 作为输入，并将其转换为一个概率分布。
            # Softmax 的输出满足两个条件：1) 每个值都在 0 和 1 之间；2) 所有值的总和为 1。
            # 这使得输出可以被解释为“输入样本属于每个类别的概率”。
            output = tf.contrib.layers.softmax(logits)

        # inference 方法返回最终的 output 张量，这个张量的形状是 (样本数, 类别数)，包含了每个样本属于各个类别的预测概率。
        return output
