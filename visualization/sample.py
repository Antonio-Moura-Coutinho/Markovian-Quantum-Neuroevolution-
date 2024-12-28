import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# 尝试用 scipy.io.loadmat 读取
vars = scipy.io.loadmat('../MNIST_1_9_wk.mat')
x_train = vars["x_train"]
y_train = vars["y_train"]
x_test = vars["x_test"]
y_test = vars["y_test"]


i = 1
a = vars["x_train"][0:256,i].real
c1 = a.reshape(16, 16)
i = 2
a = vars["x_train"][0:256,i].real
c2 = a.reshape(16, 16)
i = 4
a = vars["x_train"][0:256,i].real
c3 = a.reshape(16, 16)
i = 5
a = vars["x_train"][0:256,i].real
c4 = a.reshape(16, 16)
data = np.vstack((np.hstack((c1, c2)), np.hstack((c3, c4))))

plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()  
plt.title("Sample")
plt.savefig("Sample.pdf")