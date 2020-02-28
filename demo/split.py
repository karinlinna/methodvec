#载入需要的模块
import pandas as pd
import os

test_corpus=[]
os.getcwd()  #获取当前工作路径，查看是否是自己的目标路径
os.chdir('/Users/ljl/Downloads/glove/glove.py-master/MV_L')  #如果不是，改到目标路径
path = '/Users/ljl/Downloads/glove/glove.py-master/MV_L'
os.listdir(path) #查看目标路径下有哪些数据
datalist = []
num = 1
file_train = open('/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/train.txt', 'w')
file_val = open('/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/val.txt', 'w')
for i in os.listdir(path):
    if os.path.splitext(i)[1] == '.txt':     #选取后缀为txt的文件加入datalist
        datalist.append(i)
    df = pd.DataFrame()

for txt in datalist:
    data_path = os.path.join(path,txt)
    #f = open(data_path)  # 返回一个文件对象
    if num < 2400:
        for line in open(data_path):
            file_train.writelines(line)
    else:
        for line in open(data_path):
            file_val.writelines(line)
    # if num < 2400 :
    #     test_corpus = f.readlines()
    # else :
    #     val_corpus = f.readlines()
    num += 1
# with open('/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/train.txt', 'w') as f:
#     f.write(str(test_corpus))
# with open('/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/val.txt', 'w') as f:
#     f.write(str(val_corpus))



# #第二个版本： 获取目标文件夹的路径
# filedir = os.getcwd() + '/Users/ljl/Downloads/glove/glove.py-master/MV_L'
# # 获取当前文件夹中的文件名称列表
# filenames = os.listdir(filedir)
# # 打开当前目录下的result.txt文件，如果没有则创建
# file = open('/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/train.txt', 'w')
# # 向文件中写入字符
#
# # 先遍历文件名
# for filename in filenames:
#     filepath = meragefiledir
#     filepath = filepath + '/' +  filename
#     # 遍历单个文件，读取行数
#     for line in open(filepath):
#         file.writelines(line)
#     file.write('\n')
