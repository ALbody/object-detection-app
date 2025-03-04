from flask import Flask, Response
from flask_cors import CORS
import cv2
import os
import time

# تهيئة تطبيق Flask
app = Flask(__name__)
CORS(app)

# استخدم عنوان كاميرا الشبكة أو ملف فيديو
CAMERA_URL = os.getenv("CAMERA_URL", "video.mp4")  # يمكن تعيين URL عبر المتغيرات البيئية

# فتح اتصال الفيديو (من كاميرا IP أو ملف فيديو)
cam = cv2.VideoCapture(CAMERA_URL)

# التأكد من نجاح الاتصال
if not cam.isOpened():
    print("❌ فشل في الاتصال بالفيديو، تأكد من الرابط أو الملف!")
else:
    print("✅ الاتصال بالفيديو ناجح!")

# بث الفيديو
def generate_frames():
    global cam
    fps_limit = 10  # تحديد معدل الإطارات
    delay = 1 / fps_limit

    try:
        while True:
            start_time = time.time()
            success, frame = cam.read()
            if not success:
                print("❌ فشل في قراءة الإطار، ربما انتهى الفيديو؟")
                break

            # تحويل الصورة إلى JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elapsed_time = time.time() - start_time
            if elapsed_time < delay:
                time.sleep(delay - elapsed_time)
    finally:
        cam.release()

# تشغيل الفيديو في المتصفح
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# تغيير مصدر الفيديو ديناميكيًا
@app.route('/set_source/<source>')
def set_source(source):
    global cam, CAMERA_URL
    CAMERA_URL = source
    cam.release()
    cam = cv2.VideoCapture(CAMERA_URL)
    if not cam.isOpened():
        return f"❌ فشل في الاتصال بمصدر الفيديو: {source}", 500
    else:
        return f"✅ تم تغيير مصدر الفيديو إلى: {source}"

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
