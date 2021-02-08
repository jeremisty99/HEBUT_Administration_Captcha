import os
if __name__ == "__main__":
    path = "D:\\Code\\HEBUT\\Test\\"
    for root, dirs, files in os.walk(path):
        print(files)  # 当前路径下所有非目录子文件
    for st in files:
        if len(st) != 8:
            print(st)