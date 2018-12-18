import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas
import os
import seaborn as sns

names = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'] 

#calculate the correlation matric
iris = datasets.load_iris()
x1 = np.array([x[0] for x in iris.data])
x2 = np.array([x[1] for x in iris.data])
x3 = np.array([x[2] for x in iris.data])
x4 = np.array([x[3] for x in iris.data])

#

image = abs(np.corrcoef([x1,x2,x3,x4]))

print (np.corrcoef([x1,x2,x3,x4]))

# plot correlation matrix 
#fig = plt.figure() 
f, ax = plt.subplots(figsize=(9, 6))
#ax = fig.add_subplot(figsize=(9,6)) #图片大小为20*20
ax = sns.heatmap(image,ax=ax, linewidths=0.5,vmax=1, 
	vmin=0 ,annot=True,annot_kws={'size':10,'weight':'bold'})
#热力图参数设置（相关系数矩阵，颜色，每个值间隔等）
#ticks = numpy.arange(0,16,1) #生成0-16，步长为1 
plt.xticks(np.arange(4)+0.5,names) #横坐标标注点
plt.yticks(np.arange(4)+0.5,names) #纵坐标标注点

label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')


ax.set_xticklabels(names) #生成x轴标签 
ax.set_yticklabels(names)
ax.set_title('Iris correlation matrix')#标题设置
#plt.savefig('cluster.tif',dpi=300)
plt.show()








