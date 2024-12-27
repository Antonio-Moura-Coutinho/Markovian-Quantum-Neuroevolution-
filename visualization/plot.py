import numpy as np 
from matplotlib import pyplot as plt 

gen = np.full((7, 10), np.arange(1,8).reshape(7,1))

accs = [[0.39, 0.39, 0.39, 0.55, 0.76, 0.84, 0.555, 0.72, 0.65, 0.39],
        [0.855, 0.76, 0.775, 0.72, 0.825, 0.68, 0.775, 0.77, 0.825, 0.655],
        [0.83, 0.875, 0.965, 0.55, 0.675, 0.695, 0.735, 0.69, 0.82, 0.835],
        [0.915, 0.965, 0.91, 0.955, 0.88, 0.97, 0.955, 0.955, 0.955, 0.84],
        [0.915, 0.815, 0.9, 0.955, 0.965, 0.935, 0.905, 0.975, 0.83, 0.88],
        [0.96, 0.945, 0.945, 0.955, 0.96, 0.755, 0.95, 0.935, 0.935, 0.95],
        [0.94, 0.945, 0.955, 0.975, 0.895, 0.96, 0.895, 0.915, 0.96, 0.97]]

plt.scatter(gen, accs)

plt.title('Basic Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()