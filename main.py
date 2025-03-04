from flask import Flask, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import os

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

# متغير الكاميرا
cam = None

# قائمة المحاولات لفتح الكاميرا
camera_indexes = [0, -1, 1, 2]

def open_camera():
    """ يحاول فتح الكاميرا بعدة طرق مختلفة """
    global cam
    for index in camera_indexes:
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            print(f"✅ تم فتح الكاميرا بنجاح على index {index}")
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)
            return
        else:
            print(f"❌ فشل في فتح الكاميرا على index {index}")
    
    # إذا لم تنجح أي طريقة، يتم تعيين الكاميرا إلى None
    cam = None
    print("⚠️ لم يتم العثور على كاميرا متاحة!")

# تشغيل الكاميرا في Thread مستقل
camera_thread = threading.Thread(target=open_camera)
camera_thread.start()

# وظيفة بث الفيديو
def generate_frames():
    global cam
    while cam is None:
        print("⏳ في انتظار فتح الكاميرا...")
        time.sleep(1)  # انتظار حتى يتم فتح الكاميرا

    while True:
        success, frame = cam.read()
        if not success:
            print("❌ فشل في قراءة الإطار")
            break

        # تشغيل نموذج الكشف عن الكائنات
        class_ids, confs, bbox = net.detect(frame, confThreshold=0.5)

        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                label = class_names[class_id - 1]
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # تحويل الصورة إلى JPEG
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
    app.run(host="0.0.0.0", port=5000, debug=True)
