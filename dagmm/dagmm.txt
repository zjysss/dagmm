import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from dagmm.compression_net import CompressionNet
from dagmm.estimation_net import EstimationNet
from dagmm.gmm import GMM

from os import makedirs
from os.path import exists, join

class DAGMM:
    """ Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """
    # 定义两个常量，用于保存模型和数据缩放器时的标准文件名，避免在代码中写死字符串，是很好的编程习惯
    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens, comp_activation,
            est_hiddens, est_activation, est_dropout_ratio=0.5,
            minibatch_size=1024, epoch_size=100,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.005,
            normalize=True, random_seed=123):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        # --- 1. 初始化核心网络组件 ---
        # 创建一个“压缩网络”对象，传入其隐藏层结构和激活函数
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)

        # 创建一个“估计网络”对象，传入其隐藏层结构和激活函数
        self.est_net = EstimationNet(est_hiddens, est_activation)

        # 存储估计网络在训练时要使用的dropout比率
        self.est_dropout_ratio = est_dropout_ratio

        # --- 2. 初始化GMM组件 ---
        # 关键点：估计网络隐藏层的最后一个数字，定义了GMM要用多少个高斯分布来拟合数据
        n_comp = est_hiddens[-1]

        # 创建一个GMM（高斯混合模型）对象，告诉它需要拟合多少个组件
        self.gmm = GMM(n_comp)

        # --- 3. 存储训练超参数 ---
        self.minibatch_size = minibatch_size  # 次训练用的小批量数据的大小
        self.epoch_size = epoch_size          # 总共要训练多少轮
        self.learning_rate = learning_rate    # 学习率，控制模型参数更新的幅度
        self.lambda1 = lambda1                # 损失函数中“能量项”的权重
        self.lambda2 = lambda2                # 损失函数中“协方差对角线项”的权重（正则化项）



        # --- 4. 存储其他配置 ---
        self.normalize = normalize            # 是否需要对输入数据进行标准化
        self.scaler = None                    # 用来存储数据标准化工具（StandardScaler）
        self.seed = random_seed               # 随机种子，用于保证结果可复现

        # --- 5. 初始化TensorFlow相关变量 ---
        self.graph = None                     # 用来存储TensorFlow的计算图
        self.sess = None                      # 用来存储TensorFlow的会话（Session），是执行计算图的句柄

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def fit(self, x):                         # 训练模型
        """ Fit the DAGMM model according to the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape

        # --- 1. 数据预处理 ---
        # 如果设置了normalize=True
        if self.normalize:

            # 创建一个StandardScaler对象，它能计算数据的均值和标准差，并进行标准化
            self.scaler = scaler = StandardScaler()

            # fit_transform：学习x的均值和方差，并用它们来转换x，使x的均值为0，方差为1
            x = scaler.fit_transform(x)



        # --- 2. 构建TensorFlow计算图 ---
        # with tf.Graph().as_default() 创建一个全新的、干净的图，避免与其他模型冲突
        with tf.Graph().as_default() as graph:
            self.graph = graph

            # 设置随机种子，保证每次运行时权重初始化等随机操作的结果都一样
            tf.set_random_seed(self.seed)
            np.random.seed(seed=self.seed)

            # Create Placeholder
            # --- 2a. 定义数据入口 (Placeholders) ---
            # placeholder像一个“数据输入口”，在真正运行图时才把真实数据“喂”进去
            self.input = input = tf.placeholder(
                dtype=tf.float32, shape=[None, n_features])
            self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[]) # # dropout比率的入口

            # Build graph
            # --- 2b. 串联网络，定义数据流 ---
            # 1. 压缩：输入数据(input)流经压缩网络，得到低维表示z和重构数据x_dash
            z, x_dash  = self.comp_net.inference(input)

            # 2. 估计：低维表示z流经估计网络，得到每个样本属于各个高斯分布的概率gamma
            gamma = self.est_net.inference(z, drop)

            # 3. GMM拟合：根据z和gamma，计算GMM的参数（phi, mu, sigma）
            self.gmm.fit(z, gamma)

            # 4. 计算能量：根据z和GMM的参数，计算每个样本的能量（异常分数）
            energy = self.gmm.energy(z)

            # 保存重构结果，用于计算损失
            self.x_dash = x_dash

            # Loss function
            # --- 2c. 定义损失函数 (Loss Function) ---
            loss = (
                # 1. 重构损失：输入数据和重构数据的差异，希望它越小越好
                self.comp_net.reconstruction_error(input, x_dash) +
                # 2. 能量损失：希望正常样本的能量越小越好，由lambda1控制其重要性
                self.lambda1 * tf.reduce_mean(energy) +
                # 3. GMM协方差正则化：防止协方差矩阵退化（对角线元素太小），由lambda2控制
                self.lambda2 * self.gmm.cov_diag_loss())

            # Minimizer
            # --- 2d. 定义优化器 ---
            # 使用Adam优化器，它的任务是不断调整网络参数，以最小化上面的loss
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # Number of batch# --- 3. 执行训练 ---
            # 计算总共有多少个小批量
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            # Create tensorflow session and initilize
            # 创建并初始化Session
            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

            # Training# --- 3a. 训练循环 ---
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx) # 打乱数据顺序，增加训练的随机性

            for epoch in range(self.epoch_size):
                for batch in range(n_batch):
                    # 从打乱后的数据中，按顺序取出当前批次的数据
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[idx[i_start:i_end]]

                    # 运行优化器，并通过feed_dict把真实数据喂给placeholder
                    # drop: self.est_dropout_ratio 表示在训练时启用dropout
                    self.sess.run(minimizer, feed_dict={
                        input:x_batch, drop:self.est_dropout_ratio})

                # 每100轮打印一次当前模型在整个训练集上的损失值
                # 注意：在评估损失时，dropout比率设为0，表示不使用dropout
                if (epoch + 1) % 100 == 0:
                    loss_val = self.sess.run(loss, feed_dict={input:x, drop:0})
                    print(" epoch {}/{} : loss = {:.3f}".format(epoch + 1, self.epoch_size, loss_val))

            # Fix GMM parameter
            # --- 4. 训练收尾工作 ---
            # 训练结束后，固定GMM的参数（phi, mu, sigma），不再改变
            fix = self.gmm.fix_op()
            self.sess.run(fix, feed_dict={input:x, drop:0})

            # 把最终计算能量的操作保存下来，以便predict时使用
            self.energy = self.gmm.energy(z)

            # --- 5. 准备模型保存 ---
            # 使用tf.add_to_collection给关键的操作（tensor）打上“标签”
            # 这样在加载模型时，就可以通过标签名方便地找到它们
            tf.add_to_collection("save", self.input)
            tf.add_to_collection("save", self.energy)

            # 创建一个Saver对象，用于保存和加载模型的所有变量
            self.saver = tf.train.Saver()

    def predict(self, x):
        """ Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        # 安全检查：如果模型没训练过(self.sess是None)，就报错
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        # 如果训练时用了标准化，那么预测时也必须用同样的标准化
        if self.normalize:
            # 注意！这里用.transform()而不是.fit_transform()
            # 必须使用训练时学到的均值和方差来转换新数据
            x = self.scaler.transform(x)

        # 运行之前保存好的self.energy操作，把新数据x喂进去
        energies = self.sess.run(self.energy, feed_dict={self.input:x})
        return energies

    def save(self, fdir):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """

        if self.sess is None:
            raise Exception("Trained model does not exist.")

        # 如果目标文件夹不存在，就创建它
        if not exists(fdir):
            makedirs(fdir)

        # --- 保存TensorFlow模型 ---
        model_path = join(fdir, self.MODEL_FILENAME)

        # saver.save会保存计算图的结构和所有变量的值
        self.saver.save(self.sess, model_path)

        # --- 保存数据缩放器 ---
        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            # TensorFlow的Saver不能保存普通的Python对象，所以用joblib来保存scaler对象
            joblib.dump(self.scaler, scaler_path)

    def restore(self, fdir):
        """ Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = join(fdir, self.MODEL_FILENAME)
        meta_path = model_path + ".meta" # .meta文件保存了图的结构

        # 同样，在新的干净的图里进行恢复
        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.sess = tf.Session(graph=graph)
            # 1. 导入图结构
            self.saver = tf.train.import_meta_graph(meta_path)
            # 2. 将保存的变量值恢复到这张图里
            self.saver.restore(self.sess, model_path)
            # 3. 通过之前打的“标签”，把输入口和能量计算操作找回来
            self.input, self.energy = tf.get_collection("save")

        # 恢复scaler对象
        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)
