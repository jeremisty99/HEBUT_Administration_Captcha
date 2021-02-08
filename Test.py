from CNN import myDecode, test_loader, batch_size, CNN
import torch

if __name__ == '__main__':
    # 加载模型
    cnn = torch.load('./111.pth')
    # 每次加载batch_size张图片tensor
    for each in test_loader:
        # 分别取元组的第一、第二个元素作为图片和标签
        imgs, labels = each[0], each[1]
        # 转移到gpu
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        # 前向传播
        preds = cnn(imgs)
        # 将batch_size个tensor分开进行可视化和解码
        for i in range(1, batch_size):
            # 解码
            pre_str, tar_str = myDecode(preds[i], labels[i])
            # 打印
            print(str(i) + ' pre_str:{}||tar_str:{}'.format(pre_str, tar_str))