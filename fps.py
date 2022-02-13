import cv2
import numpy as np

class FPSCounter():
    """
    FPSを計測するクラス
    """

    def __init__(self, length=30):
        """
        FPSを計測するクラス

        Args:
            length (int, optional): 平均FPS値を求める際のフレーム数(default 60)
        """
        self.start_time = cv2.getTickCount()
        self.fps_list = np.zeros(length)

    def tick(self):
        """
        FPSを記録
        """

        # FPSの計測
        pres_time = cv2.getTickCount()
        diff_time = (pres_time - self.start_time)
        self.start_time = pres_time    # 次回FPS計測のための準備
        fps = cv2.getTickFrequency() / diff_time

        # FPSの記録
        self.fps_list[0:-1] = self.fps_list[1:]
        self.fps_list[-1] = fps

    def get(self):
        """
        平均FPSを取得

        Returns:
            int: 平均FPS
        """

        fps = int(np.average(self.fps_list))
        return fps
