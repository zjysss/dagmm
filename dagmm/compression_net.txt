import tensorflow as tf

# 主要功能: 这个网络将输入数据转换为一种适合“估算网络”（Estimation Network，代码中未提供）计算异常分数的表示形式。
# 输出组成: 网络的输出包含两个部分：
# 通过自编码器学习到的降维表示。
# 从重建误差中提取的特征。
class CompressionNet:
    """ Compression Network.
    This network converts the input data to the representations
    suitable for calculation of anormaly scores by "Estimation Network".

    Outputs of network consist of next 2 components:
    1) reduced low-dimensional representations learned by AutoEncoder.
    2) the features derived from reconstruction error.
    """
    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh):
        # self: 指向实例本身的引用。
        # hidden_layer_sizes: 一个整数列表，定义了编码器（Encoder）中每个隐藏层的神经元数量。解码器（Decoder）的结构会根据这个列表自动生成（层数和神经元数量相反）。
        # 例如，如果输入是 [128, 64]，那么编码器结构是 输入 -> 128 -> 64，解码器结构是 64 -> 128 -> 输出。
        # activation=tf.nn.tanh: 指定隐藏层的激活函数，默认为 tanh（双曲正切函数）。
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of the size of hidden layers.
            For example, if the sizes are [n1, n2],
            the sizes of created networks are:
            input_size -> n1 -> n2 -> n1 -> input_sizes
            (network outputs the representation of "n2" layer)
        activation : function
            activation function of hidden layer.
            the last layer uses linear function.
        """
        # 将传入的 hidden_layer_sizes 列表存储为实例的属性，以便类中的其他方法可以访问。
        self.hidden_layer_sizes = hidden_layer_sizes

        # 将传入的 activation 函数存储为实例的属性。
        self.activation = activation

   # 定义编码器部分，负责将输入数据 x 压缩到低维空间。
    def compress(self, x):
        # 获取输入张量 x 的第二个维度（特征数量），并将其存储在 self.input_size 中。这对于后续构建解码器以重建回原始维度至关重要。
        self.input_size = x.shape[1]

        # 创建一个名为 "Encoder" 的变量作用域。这有助于组织计算图，使得在 TensorBoard 等工具中可视化时，所有编码器的层和变量都会被归纳在这个命名空间下，同时也避免了变量名冲突。
        with tf.variable_scope("Encoder"):
            # 初始化一个变量 z，它的初始值是输入数据 x。z 将在网络中逐层传递并被转换。
            z = x
            n_layer = 0

            # 开始一个循环，遍历 hidden_layer_sizes 中除最后一个元素外的所有元素。编码器的中间隐藏层在这里构建。
            for size in self.hidden_layer_sizes[:-1]:
                n_layer += 1

                # 创建一个全连接层（也叫密集层）。
                # z: 该层的输入。
                # size: 该层的输出维度（神经元数量）。
                # activation=self.activation: 使用初始化时定义的激活函数（默认为 tanh）。
                # name="layer_{}".format(n_layer): 为该层命名，如 "layer_1", "layer_2" 等。
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))

            # activation function of last layer is linear
            n_layer += 1

            # 构建编码器的最后一层，即瓶颈层（bottleneck layer），它的维度是 hidden_layer_sizes 列表中的最后一个数字。
            # 这一层没有指定 activation 参数。在 tf.layers.dense 中，这等同于使用线性激活函数（即不对输出做任何非线性变换），这对于生成低维表示 z 是常见的做法。
            z = tf.layers.dense(z, self.hidden_layer_sizes[-1],
                name="layer_{}".format(n_layer))

        return z # 返回经过所有编码层后的最终输出，即压缩后的低维表示。

    # 定义解码器部分，负责将压缩后的低维表示 z 重建回原始数据空间。
    def reverse(self, z):
        # 为解码器的所有层创建一个名为 "Decoder" 的变量作用域。
        with tf.variable_scope("Decoder"):
            n_layer = 0

            # 遍历隐藏层尺寸列表。[::-1] 表示将列表反转。这构建了与编码器结构对称的解码器层。
            for size in self.hidden_layer_sizes[:-1][::-1]:
                n_layer += 1

                # 逐步构建解码器的隐藏层，将低维表示 z 逐层放大。
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))

            # activation function of last layes is linear
            n_layer += 1
            # 构建解码器的最后一层。这一层的输出维度必须是 self.input_size，即与原始输入数据的特征维度相同。同样，它使用线性激活函数。
            x_dash = tf.layers.dense(z, self.input_size,
                name="layer_{}".format(n_layer))

        return x_dash # 返回重建后的数据 x_dash。


    # 计算重建误差特征 不直接用于模型的梯度下降优化，而是用来计算两种特定的重建误差，并将它们作为新的特征。
    def loss(self, x, x_dash):

        # 定义一个辅助函数，用于计算输入张量 x 中每个样本（行）的欧几里得范数（L2范数）。
        # tf.square(x): 计算 x 中每个元素的平方。
        # tf.reduce_sum(..., axis=1): 沿着第二个维度（特征维度）求和，得到每个样本的范数的平方。
        # tf.sqrt(...): 开平方，得到最终的欧几里得范数。
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

        # Calculate Euclid norm, distance
        # 计算原始数据 x 中每个样本的范数。
        norm_x = euclid_norm(x)

        #计算重建数据 x_dash 中每个样本的范数。
        norm_x_dash = euclid_norm(x_dash)

        # 计算 x 和 x_dash 之间的欧几里得距离。
        dist_x = euclid_norm(x - x_dash)

        # 计算 x 和 x_dash 中对应样本的点积。
        dot_x = tf.reduce_sum(x * x_dash, axis=1)

        # Based on the original paper, features of reconstraction error
        # are composed of these loss functions:
        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity


        # 定义一个很小的正数，用于加在分母上，防止出现除以零的错误，增加数值稳定性。
        min_val = 1e-3

        # 计算第一个误差特征：相对欧几里得距离。它表示重建距离相对于原始数据“大小”的比例。
        loss_E = dist_x  / (norm_x + min_val)

        # 计算第二个误差特征：余弦相似度。dot_x / (norm_x * norm_x_dash) 是余弦相似度的计算公式。1.0 - ... 将其转换为余弦距离。
        # 0.5 * ... 是一个缩放因子。这个值越小，表示 x 和 x_dash 的方向越接近。
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))


        # loss_E[:,None] 和 loss_C[:,None]：将一维的 loss_E 和 loss_C 向量转换为 [n_samples, 1] 的二维张量。
        # tf.concat(..., axis=1): 将这两个 [n_samples, 1] 的张量沿第二个维度（列）拼接起来，形成一个 [n_samples, 2] 的张量，这个张量就是从重建误差中提取出的新特征。
        return tf.concat([loss_E[:,None], loss_C[:,None]], axis=1)


    # 特征提取与前向传播
    def extract_feature(self, x, x_dash, z_c):

        # 调用 loss 方法计算重建误差特征 z_r。
        z_r = self.loss(x, x_dash)

        # 将编码器输出的低维表示 z_c 和重建误差特征 z_r 拼接在一起，形成最终的、更丰富的特征向量 z。
        return tf.concat([z_c, z_r], axis=1)


    # 定义了整个网络的前向传播逻辑。
    def inference(self, x):
        """ convert input to output tensor, which is composed of
        low-dimensional representation and reconstruction error.

        Parameters
        ----------
        x : tf.Tensor shape : (n_samples, n_features)
            Input data

        Results
        -------
        z : tf.Tensor shape : (n_samples, n2 + 2)
            Result data
            Second dimension of this data is equal to
            sum of compressed representation size and
            number of loss function (=2)

        x_dash : tf.Tensor shape : (n_samples, n_features)
            Reconstructed data for calculation of
            reconstruction error.
        """

        # 为整个压缩网络（包括编码器和解码器）创建一个顶层的作用域 "CompNet"。
        with tf.variable_scope("CompNet"):
            # AutoEncoder

            # 调用 compress 方法，将输入 x 编码为低维表示 z_c。
            z_c = self.compress(x)

            # 调用 reverse 方法，将 z_c 解码为重建数据 x_dash。
            x_dash = self.reverse(z_c)

            # compose feature vector
            # 调用 extract_feature 方法，将 z_c 和根据 x、x_dash 计算出的重建误差特征拼接起来，得到最终的输出特征 z。
            z = self.extract_feature(x, x_dash, z_c)

        #返回 z: 最终的特征向量，将提供给后续的估算网络使用。
        # x_dash: 重建的数据，它需要被用来计算训练整个自编码器所用的损失。
        return z, x_dash

    # 计算训练损失
    def reconstruction_error(self, x, x_dash):

        # tf.square(x - x_dash): 计算原始数据 x 和重建数据 x_dash 之间每个元素的差的平方。
        # tf.reduce_sum(..., axis=1): 沿着特征维度求和，得到每个样本的总平方误差。
        # tf.reduce_mean(..., axis=0): 沿着样本维度求均值，得到当前批次（batch）数据的平均重建误差。
        # 这个标量值就是最终的损失，可以被 TensorFlow 的优化器用来更新网络权重。
        return tf.reduce_mean(tf.reduce_sum(
            tf.square(x - x_dash), axis=1), axis=0)
