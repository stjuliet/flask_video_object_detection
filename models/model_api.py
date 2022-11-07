from models.v3_fastest import *
from models.v4_tiny import *
from models.v5_dnn import *
from models.vx_ort import *


def v3_fastest(camera):
    while True:
        frame = camera.get_frame()
        v3_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg, 所以前端要接收的应该是图片enimcodo后的base64
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def v4_tiny(camera):
    while True:
        frame = camera.get_frame()
        v4_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def v5_dnn(camera):
    v5_net = yolov5()
    while True:
        frame = camera.get_frame()
        v5_net.v5_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def vx_ort(camera):
    while True:
        frame = camera.get_frame()
        yolox_detect(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')