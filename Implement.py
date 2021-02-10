import base64
import io
from torch import nn, optim
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import *


class CNN(torch.nn.Module):
    def __init__(self):
        # 先调用父类的__init__()方法
        super().__init__()
        # ORIGINAL-----batch_size * channels * width * height = 64*3*100*38----- #
        self.net = torch.nn.Sequential(
            # 64*3*180*60
            nn.Conv2d(3, 16, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # batch_size * channels * width * height = 64*16*50*19
            # 64*16*90*30
            nn.Conv2d(16, 64, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # batch_size * channels * width * height = 64*64*25*9
            # 64*64*45*15

            nn.Conv2d(64, 256, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 64*256*22*7

            nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            # batch_size * channels * width * height = 64*512*12*4
            # 64*512*11*3
        )

        self.fc = nn.Sequential(
            # 这里只传入channels*width*height
            # 全连接层得到的应该是最终的分类数，验证码每个位置有四种可能，符合加法原理
            nn.Linear(512 * 11 * 3, 36 * 4)
        )

    def forward(self, x):
        # 先进行卷积和池化
        x = self.net(x)
        # 将tensor展成一维
        x = x.view(-1, 512 * 11 * 3)
        # 全连接
        x = self.fc(x)
        return x

app = Flask(__name__)  # 固定写法
CORS(app)
model = torch.load('./111.pth')
model.eval()
model.cuda()

def distinguish(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = transforms.ToTensor()(image)

    # Add batch_size axis.
    image = image[None]
    img = image.cuda()
    pred = model(img)

    pred = pred.view(-1, 36)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    captcha = ''

    for pre_num in pred:
        if pre_num > 25:
            captcha += chr(pre_num + 22)
        else:
            captcha += chr(pre_num + 97)

    return captcha


# '/predict'是会影响请求的格式，可自由改名。
# 需要添加“get”方法，才能直接通过浏览器发送请求
# 请求的路径path是图片的路径，一般是在服务端本机
# 浏览器输入实例,请换自己的ip和路径：http://192.168.1.139:5005/predict?path=/home/ai004/sdg4.jpg


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global imgdata
    if request.method == 'POST':  # 接收传输的图片
        file = request.files['file']
    else:
        s = request.args.get("data").encode(encoding='utf-8')
        imgdata = base64.b64decode(s[22:])

    image = Image.open(io.BytesIO(imgdata))
    captcha = distinguish(image)
    return jsonify({'captcha': captcha})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)  # 使其他主机可以访问服务
