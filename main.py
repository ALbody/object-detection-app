from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time

# تهيئة تطبيق Flask
app = Flask(__name__)
CORS(app)

# تحميل نموذج الذكاء الاصطناعي للكشف عن الأجسام
net = cv2.dnn_DetectionModel('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# تحميل أسماء الأجسام من ملف coco.names
with open('coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# وظيفة بث الفيديو
def generate_frames():
    cam = cv2.VideoCapture(0)  # فتح الكاميرا
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ضبط العرض
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # ضبط الارتفاع
    cam.set(cv2.CAP_PROP_FPS, 30)            # ضبط معدل الإطارات

    if not cam.isOpened():
        print("❌ فشل في فتح الكاميرا")
        return

    while True:
        success, frame = cam.read()
        if not success:
            print("❌ فشل في قراءة الإطار")
            break

        # تحويل الألوان من BGR إلى RGB لمنع التشوه
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # تشغيل نموذج الكشف عن الكائنات
        class_ids, confs, bbox = net.detect(frame, confThreshold=0.5)

        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                label = class_names[class_id - 1]

                # رسم المستطيلات على الأجسام المكتشفة
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # إعادة ضبط حجم الصورة
        frame = cv2.resize(frame, (640, 480))

        # تحويل الصورة إلى JPEG لإرسالها إلى المتصفح
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

# تشغيل الفيديو في المتصفح
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
