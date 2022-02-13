import pyrealsense2 as rs

from configuration import ConfigGenerator, Config
import fps
import threading
import numpy as np
import cv2


# 設定ファイルの読み込み
config: Config = ConfigGenerator().generate()


class RealSense(threading.Thread):
    """
    RealSenseから画像を取得するクラス

    Attributes:
        depth_image: Depthフレーム
        color_image: RGBフレーム
        gray_image: グレースケール化したRGBフレーム
        canny_image: Canny法によりエッジを抽出したフレーム

    """

    def __init__(self):
        threading.Thread.__init__(self)

        # RealSenseの準備
        self.cfg = rs.config()
        self.cfg.enable_stream(
            rs.stream.color,
            config.realsense.width,
            config.realsense.height,
            rs.format.bgr8,
            config.realsense.fps,
        )
        self.cfg.enable_stream(
            rs.stream.depth,
            config.realsense.width,
            config.realsense.height,
            rs.format.z16,
            config.realsense.fps,
        )

        # 変数の初期化
        self.depth_image = np.zeros(
            (config.realsense.height,config.realsense.width), dtype=np.uint16
        )
        self.color_image = np.zeros(
            (config.realsense.height,config.realsense.width, 3), dtype=np.uint8
        )
        self.gray_image = np.zeros(
            (config.realsense.height,config.realsense.width), dtype=np.uint8
        )
        self.canny_image = np.zeros(
            (config.realsense.height,config.realsense.width), dtype=np.uint8
        )
        self.sharpness = 0
        self.ready = False

        # FPS計測の準備
        self.fps = fps.FPSCounter()

    def run(self):
        """
        RealSenseからのストリーミングを開始
        """

        # RealSenseのStreamを開始
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(self.cfg)
        #s = profile.get_device().query_sensors()[1]
        #s.set_option(rs.option.enable_auto_exposure, 0)
        #s.set_option(rs.option.exposure, 500.0)

        self.running = True

        # Alignオブジェクト生成
        align_to = rs.stream.color
        align = rs.align(align_to)

        # TODO: "RuntimeError: No device connected"発生時のエラー処理を入れる
        # TODO: FPSをconfig.realsense.fpsで設定した値に固定する処理を入れる

        while self.running:
            # FPSを計測する
            self.fps.tick()

            # DepthフレームとRGBフレームの取得
            frames = self.pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            if not self.ready:
                if np.all(color_frame != 0):
                    self.ready = True

            _depth_image = np.asanyarray(depth_frame.get_data())
            _color_image = np.asanyarray(color_frame.get_data())

            # フレームの鮮明さを評価
            _gray_image = cv2.cvtColor(_color_image, cv2.COLOR_BGR2GRAY)
            _canny_image = cv2.cv2.Canny(_gray_image, 50, 200)
            _sharpness = np.sum(_canny_image)

            # より鮮明なフレームが得られたら保持しているフレームを更新
            if _sharpness > self.sharpness:
                # TODO:sharpnessの更新
                self.sharpness = _sharpness
                self.depth_image = _depth_image
                self.color_image = _color_image
                self.gray_image = _gray_image
                self.canny_image = _canny_image
            

    def stop(self):
        """
        RealSenseからのストリーミングを停止
        """
        self.ready = False
        self.running = False

    def wait_for_ready(self):
        """
        readyがtrueになるまで待つ
        """
        while not self.ready:
            pass

    def get_images(self):
        """
        DepthフレームとRGBフレームを取得

        Returns:
            (ndarray, ndarray): Depthフレーム, RGBフレーム
        """
        self.sharpness = 0
        return (self.depth_image, self.color_image)

    def get_color_image(self):
        """
        RGBフレームを取得

        Returns:
            ndarray: RGBフレーム
        """
        self.sharpness = 0
        return self.color_image

    def get_depth_image(self):
        """
        Depthフレームを取得

        Returns:
            ndarray: RGBフレーム
        """
        self.sharpness = 0
        return self.depth_image

    def get_gray_image(self):
        """
        グレースケール化したRGBフレームを取得

        Returns:
            ndarray: グレースケール化したRGBフレーム
        """
        self.sharpness = 0
        return self.gray_image

    def get_canny_image(self):
        """
        Canny法によりエッジを抽出したフレームを取得
        Returns:
            ndarray: エッジを抽出したフレーム
        """
        self.sharpness = 0
        return self.canny_image

    def get_fps(self):
        """
        RealSenseからフレーム取得にかかったfps値を取得

        Returns:
            int: fps値
        """
        return self.fps.get()

    def __del__(self):
        self.pipeline.stop()
