import cv2
import numpy as np
import datetime

def plot_fps(fps_camera, image):
    # FPSの表示
    put_outlined_text("CAM:{:00d}".format(fps_camera), (10, 20), image)


def plot_current_time(image):
    # 現在時刻の表示
    current_time = datetime.datetime.now()
    put_outlined_text(current_time.strftime("%H:%M:%S"), (730, 20), image)
    
def put_outlined_text(text, point, image, color=(0, 255, 0)):
    # アウトライン付きの文字を描写
    cv2.putText(
        image,
        text=text,
        org=point,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text=text,
        org=point,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

