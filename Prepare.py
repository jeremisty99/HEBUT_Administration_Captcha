import os
import re

#声明图片路径
dir_train = "D:\\Code\\HEBUT\\Train\\"
dir_test = "D:\\Code\\HEBUT\\Test\\"

#列出所有图片的名字
fileDist_train = os.listdir(dir_train)
fileDist_test = os.listdir(dir_test)

train = open('./train.txt','a')                   #a为追加模式
test = open('./test.txt','a')

for file in fileDist_train:
	#取.png前的部分，作为标签，将路径和标签中间隔一个空格后逐行写入.txt文件中
	name = dir_train + file + ' ' + re.split('\.',file)[0] + '\n'
	train.write(name)

for file in fileDist_test:
	name = dir_test + file + ' ' + re.split('\.',file)[0] + '\n'
	test.write(name)

test.close()
train.close()
