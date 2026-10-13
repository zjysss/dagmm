import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from dagmm import DAGMM

# --- 1. 设置数据来源 ---
url_base = "http://kdd.ics.uci.edu/databases/kddcup99"  # 定义了数据集的下载地址。

# KDDCup 10% Data
url_data = f"{url_base}/kddcup.data_10_percent.gz"    # 10%的样本数据
# info data (column names, col types)
url_info = f"{url_base}/kddcup.names"   # 包含列名信息的文件


# Import info data
# --- 2. 读取列名和类型信息 ---
# 读取kddcup.names文件。这个文件描述了数据每一列的名称和类型（是连续的数字还是离散的文本）。
df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])

# 从刚刚读取的信息中，分别抽取出列名列表和对应的类型列表（float代表数字，str代表文本）。
colnames = df_info.colname.values
coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")

# 在最后手动加上一列，名为status，它记录了每次网络连接是“正常的”(normal)还是“异常的”(attack)。
colnames = np.append(colnames, ["status"])
coltypes = np.append(coltypes, ["str"])

# Import data
# # --- 3. 读取核心数据 ---
# 使用pandas读取主要的训练数据。names=colnames把我们刚刚准备好的列名赋给数据，dtype=...则告诉pandas每一列应该是什么数据类型，这可以防止它猜错。
df = pd.read_csv(url_data, names=colnames, index_col=False,
                 dtype=dict(zip(colnames, coltypes)))

# Dumminize
# # --- 4. 数据预处理 - 独热编码 (One-Hot Encoding) ---
# df.iloc[:,:-1]: 选取了数据的所有行，以及除了最后一列（status列）之外的所有特征列。
# pd.get_dummies(...): 这是极其关键的一步。原始数据中有很多文本类型的列（比如协议类型：'tcp', 'udp', 'icmp'）。
# 机器学习模型无法理解文本，必须将它们转换成数字。
# get_dummies做的事情就是独热编码：它会把protocol这一列，变成protocol_tcp, protocol_udp, protocol_icmp三列。
# 如果某一行是'tcp'，那么在protocol_tcp这一列就是1，其他两列是0。这样，所有的文本信息就被转化成了模型可以理解的0和1。
# .values: 将处理好的pandas数据框（DataFrame）转换成numpy数组，这是模型需要的标准输入格式。
X = pd.get_dummies(df.iloc[:,:-1]).values

# Create Traget Flag
# Anomaly data when status is normal, Otherwise, Not anomaly.

# # --- 5. 创建目标标签 ---
# df.status == "normal.": 判断status列的值是否为 "normal."。
# np.where(..., 1, 0): 如果条件为真（即是"normal."），则y对应的值为 1；如果为假（即是攻击），则y对应的值为 0。
# 所以，我们最终定义：1 代表正常，0 代表异常。
y = np.where(df.status == "normal.", 1, 0)


# Split Data
# 划分训练集和测试集
# --- 1. 正常划分 ---
# 将数据X和标签y按照50:50的比例划分成训练集 (X_train, y_train) 和测试集 (X_test, y_test)。random_state确保每次划分结果都一样，便于复现。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=123)

# # --- 2. 关键步骤：只用正常数据进行训练！---
# 这是无监督异常检测的精髓所在！
# y_train == 0 会生成一个布尔索引，只有在y_train值为0（即“正常”）的位置才是True。
# X_train[y_train == 0]: 利用这个索引，我们从X_train中只把所有“正常”的样本筛选出来，作为最终的训练集。
# 目的： 我们只给模型看“好人”长什么样。通过学习大量“好人”的特征，模型就能建立一个关于“什么是正常”的深刻理解。
# 之后遇到一个“坏人”（异常数据）时，模型就会发现它和自己见过的所有“好人”都不一样，从而给出一个很高的异常分数。
X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]


# 构建并训练DAGMM模型
# model = DAGMM(...): 初始化DAGMM模型，并设置一系列超参数：
# comp_hiddens: 定义了压缩网络的结构。这里是 输入层 -> 60 -> 30 -> 10 -> 1（瓶颈层）。
# est_hiddens: 定义了估计网络的结构。这里是 瓶颈层输出 -> 10 -> 4。最后一个数字 4 表示GMM有4个高斯组件。
# learning_rate, epoch_size (训练轮数), minibatch_size (每批次大小) 等都是标准的训练参数。
# model.fit(X_train): 最重要的一步，用我们准备好的、纯净的正常数据来训练模型。
model = DAGMM(
    comp_hiddens=[60, 30, 10, 1], comp_activation=tf.nn.tanh,
    est_hiddens=[10, 4], est_dropout_ratio=0.5, est_activation=tf.nn.tanh,
    learning_rate=0.0001, epoch_size=200, minibatch_size=1024, random_seed=1111
)

model.fit(X_train)


# 评估模型性能
# 用训练好的模型来对测试集（包含正常和异常数据）进行预测。
# 注意：y_pred 在这里不是直接输出0或1，而是为每个测试样本计算出的**“能量”（Energy），也就是异常分数**。分数越高，代表模型认为它越可疑。
y_pred = model.predict(X_test)

# Energy thleshold to detect anomaly = 80% percentile of energies

# # --- 设定异常阈值 ---
# 我们如何从连续的能量分数得到0/1的判断呢？需要一个阈值。
# 这里的策略是：取所有测试样本能量分数的第80个百分位作为阈值。这意味着，我们把能量分数排在前20%的样本都划分为“异常”。这是一个经验性的选择，可以根据实际需求调整。
anomaly_energy_threshold = np.percentile(y_pred, 80)
print(f"Energy thleshold to detect anomaly : {anomaly_energy_threshold:.3f}")


# Detect anomalies from test data
# # --- 根据阈值进行最终判断 ---
# 将能量分数和阈值进行比较。如果一个样本的能量大于等于阈值，就标记为1（异常），否则标记为0（正常）。现在我们得到了模型最终的0/1判断结果 y_pred_flag。
y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)

# --- 计算性能指标 ---
# precision_recall_fscore_support(y_test, y_pred_flag, ...): 这里我们将模型的预测 y_pred_flag 和真实的标签 y_test 进行比较。
# Precision (精确率): 在所有被模型预测为异常的样本中，有多少是真正异常的？（衡量“抓得准不准”）
# Recall (召回率): 在所有真正是异常的样本中，有多少被模型成功抓到了？（衡量“抓得全不全”）
# F1-Score: 精确率和召回率的调和平均数，一个综合性的指标。
prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
print(f" Precision = {prec:.3f}")
print(f" Recall    = {recall:.3f}")
print(f" F1-Score  = {fscore:.3f}")