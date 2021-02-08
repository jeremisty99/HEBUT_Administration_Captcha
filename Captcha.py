import base64
import json
import shutil

import requests
import os

def base64_api(uname, pwd, img):
    with open(img, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        b64 = base64_data.decode()
    data = {"username": uname, "password": pwd, "image": b64}
    result = json.loads(requests.post("http://api.ttshitu.com/base64", json=data).text)
    if result['success']:
        return result["data"]["result"]
    else:
        return result["message"]
    return ""


if __name__ == "__main__":
    for i in range(406, 500):
        img_path = "D:\\Code\\HEBUT\\Captcha2\\captcha(" + str(i) + ").jpg"
        result = base64_api(uname='865957991', pwd='991229abc', img=img_path)
        new_path = "D:\\Code\\HEBUT\\Test\\" + result + '.jpg'
        if os.path.exists(new_path):
            print(result + "已存在")
            continue
        # 不存在则重命名
        shutil.copy(img_path, new_path)
        print("完成" + str(i))
        # os.rename(img_path, newName)
