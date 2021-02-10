# HEBUT_Administration_Captcha

## **河北工业大学(HEBUT)教务处网站验证码识别**

Notice: 本项目仅为个人学习记录所用 存在相当多的问题 不得用于非法用途

1. Captcha.py:  通过打码平台对爬取的验证码图片进行标注 分为训练集和测试集

2. CNN.py: 使用Pytorch训练卷积神经网络模型用来实现验证码识别

   主要参考 https://blog.csdn.net/namespace_pt/article/details/104488258 这篇文章

3. Implement.py: 使用Flask将训练好的模型部署上线创建接口

4. script.js: 油猴js脚本实现自动填充


