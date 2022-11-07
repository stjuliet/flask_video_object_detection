# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import time
import argparse
from datetime import timedelta
from models.model_api import v3_fastest, v4_tiny, v5_dnn, vx_ort


# 上传文件路径
upload_path = ""
# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp', 'mp4', 'ts', 'avi'}


class Video(object):
    """
    传入视频路径，实时播放视频
    """
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_frame(upload_path):
    camera = cv2.VideoCapture(upload_path)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
    i = 1
    while True:
        retval, im = camera.read()
        if retval:
            im = cv2.resize(im, size)
            img_code = cv2.imencode('.jpg',im)[1]
            string_data = img_code.tostring()
            yield b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+string_data+b'\r\n'
            i += 1
        else:
            camera.release()


app = Flask(__name__)
# 设置文件过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # 主路径入口必须单斜杠
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp、ts"})

        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        global upload_path
        if not os.path.exists('static/images'):
            os.makedirs('static/images')
        upload_path = os.path.join(base_path, 'static/images', f.filename)
        if not os.path.exists(upload_path):
            f.save(upload_path)  # 如果路径下不存在该文件，则保存在规定路径，否则会出错
        return render_template('upload_ok_video.html', val1=time.time())
    return render_template('upload1_video.html')


@app.route('/video_feed')
def video_feed():
    # 检测
    if args.det:
        if args.model == 'v3_fastest':
            return Response(v3_fastest(Video(upload_path)), mimetype='multipart/x-mixed-replace; boundary=frame')
        if args.model == 'v4_tiny':
            return Response(v4_tiny(Video(upload_path)), mimetype='multipart/x-mixed-replace; boundary=frame')
        if args.model == 'v5_dnn':
            return Response(v5_dnn(Video(upload_path)), mimetype='multipart/x-mixed-replace; boundary=frame')
        if args.model == 'vx_ort':
            return Response(vx_ort(Video(upload_path)), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # 显示视频，不做检测
        return Response(get_frame(upload_path), mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--model', type=str, default='v3_fastest', choices=['v3_fastest', 'v4_tiny', 'v5_dnn', 'vx_ort'])
    parser.add_argument('--det', type=bool, default=True)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = parse_args()

    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)
