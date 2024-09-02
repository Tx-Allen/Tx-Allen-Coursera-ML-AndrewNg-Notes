import numpy as np 
import matplotlib.pyplot as plt
# 设置图表样式 
# plt.style.use('./deeplearning.mplstyle')

# This lab will use a simple data set with only two data points

# 创建数据集，x为输入特征，y为目标变量
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# Number of training examples m

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
#.shape输出（样本数，特征数），.shape[0]输出样本数
m = x_train.shape[0]
# Second way to realize, len()返回数组的第一维度的长度
# m = len(x_train)

# Training example x_i, y_i，查看数据集中样本对

i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plotting the data

# Plot the data points，marker参数指定每个点的标记样式为x，c 参数用于指定点的颜色。
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

# Model Function

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

# 构造线性回归函数
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    #np.zeros(m) 会创建一个长度为 m 的一维数组，数组中的每个元素都初始化为 0。避免在后续代码中因访问未初始化的数组元素而导致错误。
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# 显示图例，将预测值和实际值的标记与其标签相关联，方便阅读图表。（左上角）
plt.legend()
plt.show()
 
x_i = 1.2
cost_1200sqft = w * x_i + b    
print(f"${cost_1200sqft:.0f} thousand dollars")
