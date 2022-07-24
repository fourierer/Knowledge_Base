import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.reshape.reshape import stack
plt.close('all')

####################### Basic plotting ############################
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
# print(ts)
ts = ts.cumsum()
# print(ts)
# ts.plot()
# 需要plt.show()可视化


df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list("ABCD"))
df = df.cumsum()
# df.plot()



####################### bar plot ############################
df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
df.plot.bar()
df.plot.bar(stacked=True)
df.plot.barh()
plt.show()

