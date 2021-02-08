import io
from CNN import CNN
import torch
from PIL import Image
from torchvision.transforms import transforms

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


if __name__ == '__main__':
    file = open("D:\\Code\\HEBUT\\Captcha\\captcha (5).jpg", 'rb')
    image = Image.open(io.BytesIO(file.read()))
    captcha = distinguish(image)
    print(captcha)