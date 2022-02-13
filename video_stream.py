import cv2
import threading
import numpy as np

from configuration import ConfigGenerator, Config
from realsense import RealSense

from flask import Flask, render_template, Response

# 設定ファイルの読み込み
config: Config = ConfigGenerator().generate()

# グローバル変数の定義
streaming_image = np.zeros(
    (config.realsense.height, config.realsense.width, 3), dtype=np.uint8
)
app = Flask(__name__)

# Flaskの設定
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def gen():
    while True:
        _, jpeg = cv2.imencode(".jpg", streaming_image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
        )

if __name__ == "__main__":
    # RealSenseの準備
    realsense = RealSense()
    realsense.start()
    realsense_warmup = threading.Thread(target=realsense.wait_for_ready())
    realsense_warmup.start()

    # ウォームアップ完了を待つ
    realsense_warmup.join()

    # Flaskの起動
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=config.debug.streaming_port,
            debug=False,
        ),
        daemon=True,
    ).start()

    try:
        while True:
            # DepthフレームとRGBフレームの取得
            depth_image, color_image = realsense.get_images()

            # Depthフレームのカラーマップを作成
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET
            )

            streaming_image = np.hstack((color_image, depth_colormap))

            cv2.imshow("Result", streaming_image)
            cv2.waitKey(1)
    finally:
        realsense.stop()
        realsense.join()
        # flask.join()
        cv2.destroyAllWindows()
