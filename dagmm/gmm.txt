# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

#  GMM 类用于对数据进行概率建模，它假设数据点是由多个高斯分布（即“钟形曲线”）混合生成的。
#  在异常检测的场景下，它通常被用在 CompressionNet 之后，来对压缩后的特征空间（z）进行密度估计。
#  正常点应该落在高概率密度的区域，而异常点则会落在低概率密度的区域。
class GMM:
    """ Gaussian Mixture Model (GMM) """

    # def __init__(self, n_comp):: 类的构造函数。
    # n_comp: 一个整数，代表高斯混合模型中混合成分（高斯分布）的数量。
    # self.n_comp = n_comp: 将混合成分的数量存储为实例的属性。
    # self.phi = self.mu = self.sigma = None: 初始化 GMM 的三个核心参数：
    # phi: 混合系数（每个高斯分布的权重），初始为 None。
    # mu: 每个高斯分布的均值，初始为 None。
    # sigma: 每个高斯分布的协方差矩阵，初始为 None。
    # self.training = False: 一个布尔标志，用于跟踪模型当前是否处于训练状态。
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.phi = self.mu = self.sigma = None
        self.training = False

    # 定义一个方法，用来创建 GMM 参数的 TensorFlow 变量。这些是持久化的、可在训练中被修改的张量。
    # n_features: 输入数据的特征维度。
    def create_variables(self, n_features):

        # 创建一个名为 "GMM" 的变量作用域，以组织计算图中的变量。
        with tf.variable_scope("GMM"):
            # 创建一个名为 "phi" 的变量，形状为 [n_comp]，用于存储混合系数。
            phi = tf.Variable(tf.zeros(shape=[self.n_comp]),
                dtype=tf.float32, name="phi")
           # 创建一个名为 "mu" 的变量，形状为 [n_comp, n_features]，用于存储 n_comp 个高斯分布的均值。
            mu = tf.Variable(tf.zeros(shape=[self.n_comp, n_features]),
                dtype=tf.float32, name="mu")
            # 创建一个名为 "sigma" 的变量，形状为 [n_comp, n_features, n_features]，用于存储 n_comp 个协方差矩阵。
            sigma = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="sigma")
            # 创建一个名为 "L" 的变量，形状与 sigma 相同。它将用于存储协方差矩阵 sigma 的 Cholesky 分解结果。
            # Cholesky 分解是一种数值上更稳定的处理协方差矩阵的方法。
            L = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="L")

        # 返回创建的所有变量。
        return phi, mu, sigma, L

    # 拟合 GMM 参数
    # 期望最大化（EM）算法中的 M-step（最大化步骤）。它假设 E-step（期望步骤）已经在别处完成，其结果 gamma 被作为输入。
    # 定义 GMM 的参数更新/拟合方法。
    # z: 输入数据，形状为 (n_samples, n_features)。
    # gamma: 责任（responsibility）矩阵，形状为 (n_samples, n_comp)。gamma[i, k] 表示数据点 i 属于第 k 个高斯分布的概率。
    def fit(self, z, gamma):
        """ fit data to GMM model

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data fitted to GMM.
        gamma : tf.Tensor, shape (n_samples, n_comp)
            probability. each row is correspond to row of z.
        """

        with tf.variable_scope("GMM"):
            # Calculate mu, sigma
            # i   : index of samples
            # k   : index of components
            # l,m : index of features
           # 对 gamma 按列求和，得到每个高斯分量所“负责”的样本总权重 N_k。
            gamma_sum = tf.reduce_sum(gamma, axis=0)

            # 更新混合系数 phi。phi_k = N_k / N，其中 N 是总样本数。这正是对 gamma 按列求均值。
            self.phi = phi = tf.reduce_mean(gamma, axis=0)

            # 更新均值 mu。tf.einsum 执行爱因斯坦求和约定。
            # 'ik,il->kl': 表示将 gamma (ik) 和 z (il) 按样本索引 i 进行加权求和，得到每个分量 k 和每个特征 l 的加权和。
            # / gamma_sum[:,None]: 将结果除以 N_k，得到每个分量的加权平均值，即新的均值 mu_k。 [:,None] 用于广播，使维度匹配。
            self.mu = mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]

            # 计算中心化的 z，并用 gamma 的平方根进行加权。这是为了下一步高效地计算协方差矩阵。
            z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - mu[None,:,:])

            # 更新协方差矩阵 sigma。
            # 'ikl,ikm->klm': 对 z_centered 执行外积操作并求和，计算出每个分量的加权协方差。
            # / gamma_sum[:,None,None]: 除以 N_k，完成协方差的计算。
            self.sigma = sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]

            # Calculate a cholesky decomposition of covariance in advance
            n_features = z.shape[1]

            # 创建一个对角矩阵，对角线上是非常小的正数 (1e-6)。
            min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6

            # 计算协方差矩阵 sigma 的 Cholesky 分解。加上 min_vals 是为了确保 sigma 是正定的，这是 Cholesky 分解的必要条件，从而增强了计算的数值稳定性。
            self.L = tf.cholesky(sigma + min_vals[None,:,:])

        self.training = False

    # 将 fit 方法计算出的动态张量（每次运行时值都可能不同）的值，赋给 create_variables 创建的静态 tf.Variable，从而“冻结”模型参数，以便在推理阶段使用。
    # 定义一个返回“固定操作”的方法。
    def fix_op(self):
        """ return operator to fix paramters of GMM
        Using this operator outside of this class,
        you can fix current parameter to static tensor variable.

        After you call this method, you have to run result
        operator immediatelly, and call energy() to use static
        variables of model parameter.

        Returns
        -------
        op : operator of tensorflow
            operator to assign current parameter to variables
        """

        # 创建一组新的 tf.Variable。
        phi, mu, sigma, L = self.create_variables(self.mu.shape[1])

        # 创建一个 TensorFlow 操作 op。
        # tf.assign 是一个操作，当在会话中运行时，它会把第二个参数（self.phi，即 fit 中计算出的张量）的值赋给第一个参数（phi，即新创建的变量）。
        # tf.group 将所有这些赋值操作捆绑在一起。
        op = tf.group(
            tf.assign(phi, self.phi),
            tf.assign(mu, self.mu),
            tf.assign(sigma, self.sigma),
            tf.assign(L, self.L)
        )

        # 这是一个关键的“指针”交换。它将实例属性 self.phi 指向新创建的、可持久化的 tf.Variable phi。
        # 而原始的、在 fit 中计算的动态张量则被保存在 self.phi_org 中。这使得后续调用 energy() 方法时，会使用这些被“固定”下来的参数值。
        self.phi, self.phi_org = phi, self.phi
        self.mu, self.mu_org = mu, self.mu
        self.sigma, self.sigma_org = sigma, self.sigma
        self.L, self.L_org = L, self.L

        self.training = False

        # 返回这个赋值操作。需要在一个 tf.Session 中 run 这个 op，才能真正完成参数的固定。
        return op

    #  定义计算能量的方法。
    # 在概率模型中，“能量”通常与负对数似然（negative log-likelihood）成正比。能量越低，表示数据点属于该模型的概率越高。
    def energy(self, z):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        energy : tf.Tensor, shape (n_samples)
            calculated energies
        """

        if self.training and self.phi is None:
            self.phi, self.mu, self.sigma, self.L = self.create_variable(z.shape[1])

        with tf.variable_scope("GMM_energy"):
            # Instead of inverse covariance matrix, exploit cholesky decomposition
            # for stability of calculation.

           # 计算每个数据点 z 相对于每个高斯分量均值 mu_k 的差值。
            z_centered = z[:,None,:] - self.mu[None,:,:]  #ikl

            # 它利用 Cholesky 分解 L 来计算马氏距离中的 (z-mu)^T * Sigma^-1 * (z-mu) 部分，但避免了直接求 Sigma 的逆（这是一个数值不稳定的操作）。
            v = tf.matrix_triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

            # log(det(Sigma)) = 2 * sum[log(diag(L))]
            # 高效地计算协方差矩阵行列式的对数。利用了性质 log(det(Sigma)) = log(det(L*L^T)) = 2 * log(det(L))，而三角矩阵 L 的行列式等于其对角线元素的乘积。
            log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)), axis=1)

            # To calculate energies, use "log-sum-exp" (different from orginal paper)
            d = z.get_shape().as_list()[1]

            # 计算每个数据点在每个高斯分量下的对数概率密度，再乘以其权重。这对应于 log(phi_k * N(z | mu_k, sigma_k))。
            logits = tf.log(self.phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.log(2.0 * np.pi) + log_det_sigma[:,None])

            # tf.reduce_logsumexp 是一个在数值上稳定的计算 log(sum(exp(x))) 的函数。
            # 这里它被用来计算混合模型的总对数似然 log(sum_k(phi_k * N(z | mu_k, sigma_k)))。取负号后，就得到了能量值。
            energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    # 一个正则化项，用于防止协方差矩阵“坍缩”（即对角线元素变得过小），这会导致数值不稳定。
    # 定义计算该损失的方法。
    def cov_diag_loss(self):

        # tf.matrix_diag_part(self.sigma): 提取所有协方差矩阵的对角线元素（即每个特征的方差）。
        # tf.divide(1, ...): 计算这些方差的倒数。
        # tf.reduce_sum(...): 将所有倒数相加。如果某个方差变得非常接近于零，它的倒数就会变得非常大，从而导致这个损失项急剧增大，以此来惩罚这种情况。
        with tf.variable_scope("GMM_diag_loss"):
            diag_loss = tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(self.sigma)))

        return diag_loss
