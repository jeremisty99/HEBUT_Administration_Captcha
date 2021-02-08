// ==UserScript==
// @name         教务处验证码填写
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        http://jwxs.hebut.edu.cn/login
// @grant        none
// @require      https://lib.baomitu.com/jquery/1.12.4/jquery.min.js
// ==/UserScript==

(function () {
    setTimeout(function () {
        img = $("#captchaImg")[0];
        var canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
        var dataURL = canvas.toDataURL("image/png");
        $.ajax({
            url: "http://192.168.123.100:5005/predict",
            dataType: "json",
            type: "get",
            data: {
                data: dataURL
            },
            success: function (res) {
                console.log(res.captcha)
                $("#input_checkcode")[0].value = res.captcha
            }
        });
    }, 1000);
})();