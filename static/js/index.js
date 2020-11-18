var canvas = document.getElementById('canvas')
// 画板大小
canvas.width = 280
canvas.height = 280

var context = canvas.getContext('2d')
// 背景颜色
context.fillStyle = 'black'
context.fillRect(0, 0, canvas.width, canvas.height)

// 线宽提示
var range = document.getElementById('customRange1')
range.oninput = function () {
    this.title = 'lineWidth: ' + this.value
}

var Mouse = { x: 0, y: 0 }
var lastMouse = { x: 0, y: 0 }
var painting = false

canvas.onmousedown = function () {
    painting = !painting
}

canvas.onmousemove = function (e) {
    lastMouse.x = Mouse.x
    lastMouse.y = Mouse.y
    Mouse.x = e.pageX - this.offsetLeft
    Mouse.y = e.pageY - this.offsetTop
    if (painting) {
        // 画笔参数：
        // linewidth: 线宽
        // lineJoin: 线条转角的样式, 'round': 转角是圆头
        // lineCap: 线条端点的样式, 'round': 线的端点多出一个圆弧
        // strokeStyle: 描边的样式, 'white': 设置描边为白色
        context.lineWidth = range.value
        context.lineJoin = 'round'
        context.lineCap = 'round'
        context.strokeStyle = 'white'

        // 开始绘画
        context.beginPath()
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath()
        context.stroke()
    }
}

canvas.onmouseup = function () {
    painting = !painting
}

// 预测图片
var predict = document.getElementById('predict')
predict.onclick = function () {
    var canvas = document.getElementById('canvas')
    imgUrl = canvas.toDataURL('image/jpeg')

    // 下载图片
    // var a = document.createElement('a')
    // a.download = './canvas'
    // a.href = imgUrl
    // document.body.appendChild(a)
    // a.click()
    // document.body.removeChild(a)

    // 发送请求
    $.ajax({
        type: 'POST',
        url: '/predict/',
        data: imgUrl,
        success: function (r) {
            $('#result').text('Prediction: ' + r.prediction)
            $('#result').attr('data-original-title', 'Confidence: ' + r.confidence)
        },
        error: function (e) {
            console.log(e)
        }
    })
}

// 清空画布
var clear = document.getElementById('clear')
clear.onclick = function () {
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.fillStyle = 'black'
    context.fillRect(0, 0, canvas.width, canvas.height)
    $('#result').html('Prediction:&nbsp;&nbsp;&nbsp;')
    $('#result').attr('data-original-title', '')
}

// 移动端实现
var start_x, start_y, move_x, move_y, end_x, end_y;

// 按下
canvas.ontouchstart = function (e) {
    start_x = e.touches[0].pageX - this.offsetLeft;
    start_y = e.touches[0].pageY - this.offsetTop;
    context.lineWidth = range.value
    context.lineJoin = 'round'
    context.lineCap = 'round'
    context.strokeStyle = 'white'
    context.beginPath();
    context.moveTo(start_x, start_y);
};

// 移动
canvas.ontouchmove = function (e) {
    move_x = e.touches[0].pageX - this.offsetLeft;
    move_y = e.touches[0].pageY - this.offsetTop;
    context.lineTo(move_x, move_y);
    context.stroke();
};

// 松开
canvas.ontouchend = function (e) {
    end_x = e.changedTouches[0].pageX - this.offsetLeft;
    end_y = e.changedTouches[0].pageY - this.offsetTop;
    context.closePath();
}

// 开启bootstrap的冒泡提示
$(function () {
    $('[data-toggle="tooltip"]').tooltip()
})