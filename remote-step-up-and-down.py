import pyrealsense2 as rs
import numpy as np
import cv2
import fps
import utils
import threading
import time
from configuration import ConfigGenerator, Config
import video_stream as vs
from realsense import RealSense

# 設定ファイルの読み込み
config: Config = ConfigGenerator().generate()

# RealSenseの準備
realsense = RealSense()
realsense.start()
realsense_warmup = threading.Thread(target=realsense.wait_for_ready())
realsense_warmup.start()

# TODO: ウォームアップ作業が並列に実行されているかを確認する

# ウォームアップ完了を待つ
realsense_warmup.join()

# FPS計測の準備
fps = fps.FPSCounter()

# ROI領域の設定(left, top, right, bottom)
roi = {
    "left": int((config.realsense.width - config.roi.width) / 2),
    "right": int((config.realsense.width + config.roi.width) / 2),
    "top": int((config.realsense.height - config.roi.height) / 2),
    "bottom": int((config.realsense.height + config.roi.height) / 2),
}

# 結果の動画保存の準備
if config.debug.save_movie_enable:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        config.debug.movie_folder_path + "/remote_step_up_and_down.avi",
        fourcc,
        config.realsense.fps,
        (config.realsense.width, config.realsense.height * 2),
    )
    
# ストリーミング配信の準備
if config.debug.streaming_enable:
    threading.Thread(
        target=lambda: vs.app.run(
            host="0.0.0.0",
            port=config.debug.streaming_port,
            debug=False,
        ),
        daemon=True,
    ).start()

try:
    while True:
        # FPSを計測する
        fps.tick()

        # DepthフレームとRGBフレームの取得
        depth_image, color_image = realsense.get_images()

        # ROI領域の抽出
        roi_depth = depth_image[roi["top"] : roi["bottom"], roi["left"] : roi["right"]]

        # Depth値の和の分布を表すグラフ
        depth_h = np.zeros_like(color_image)
        depth_v = np.zeros_like(color_image)

        def conv_depth_x(depth_x):
            result = (
                config.realsense.height
                - depth_x * config.realsense.height / config.depth_max
            )
            return result.astype(int)

        def conv_depth_y(depth_y):
            result = (
                config.realsense.width
                - depth_y * config.realsense.width / config.depth_max
            )
            return result.astype(int)

        # Height方向へDepth値の和をとったグラフを作成する関数
        def plot_depth_h(depth_h, left, right, color, thickness=2):
            depth_x = depth_image[roi["top"] : roi["bottom"], left:right].mean(axis=0)
            point_x = np.arange(left, right)
            point_y = conv_depth_x(depth_x)
            points = np.vstack([point_x, point_y]).T.reshape(1, -1, 2)
            cv2.polylines(
                depth_h, points, isClosed=False, color=color, thickness=thickness
            )

        # Width方向へDepth値の和をとったグラフを作成する関数
        def plot_depth_v(depth_v, top, bottom, color, thickness=2):
            depth_y = depth_image[top:bottom, roi["left"] : roi["right"]].mean(axis=1)
            point_x = conv_depth_y(depth_y)
            point_y = np.arange(top, bottom)
            points = np.vstack([point_x, point_y]).T.reshape(1, -1, 2)
            cv2.polylines(
                depth_v, points, isClosed=False, color=color, thickness=thickness
            )

        # ROI領域内の結果をプロット
        plot_depth_h(depth_h, roi["left"], roi["right"], (0, 255, 0))
        plot_depth_v(depth_v, roi["top"], roi["bottom"], (0, 255, 0))

        # ROI領域外の結果をプロット
        plot_depth_h(depth_h, 0, roi["left"], (255, 255, 255))
        plot_depth_h(depth_h, roi["right"], config.realsense.width, (255, 255, 255))
        plot_depth_v(depth_v, 0, roi["top"], (255, 255, 255))
        plot_depth_v(depth_v, roi["bottom"], config.realsense.height, (255, 255, 255))

        # Hough変換でライン検出
        """
        gray = cv2.cvtColor(depth_h, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(
            gray,
            rho=1,
            theta=np.pi / 360,
            threshold=20,
            minLineLength=500,
            maxLineGap=200,
        )
        if lines is not None:
            x1, y1, x2, y2 = lines[0][0]
            cv2.line(depth_h, (x1, y1), (x2, y2), (0, 0, 255), 3)
            angle = np.rad2deg(np.abs(np.arctan2((y2 - y1), (x2 - x1))))
            print(angle, "deg")
        """

        # 最小二乗法で1次関数の傾きと切片を求める
        def lsm(x, y):
            n = len(x)
            a = (np.dot(x, y) - y.sum() * x.sum() / n) / (
                (x**2).sum() - x.sum() ** 2 / n
            )
            b = (y.sum() - a * x.sum()) / n
            return a, b

        # 壁との相対ヨー角の近似値を求める
        a_h, b_h = lsm(
            np.arange(roi["left"], roi["right"]),
            roi_depth.mean(axis=0),
        )
        yaw_angle = np.rad2deg(np.arctan(a_h))

        cv2.line(
            depth_h,
            (0, conv_depth_x(b_h)),
            (config.realsense.width, conv_depth_x(b_h + config.realsense.width * a_h)),
            (0, 0, 255),
            1,
        )

        # 壁との距離を求める
        # TODO:ピッチ角を考慮して距離の補正を入れる
        distance = roi_depth.mean(dtype=int)

        info_yaw_angle = {
            "value": yaw_angle,
            "text": "YAW ANGLE:{:+.2f}deg   ".format(yaw_angle),
            "color": (0, 0, 0),
        }
        info_distance = {
            "value": distance,
            "text": "DISTANCE: {:+4d}mm  ".format(distance),
            "color": (0, 0, 0),
        }

        if yaw_angle > config.angle_max:
            info_yaw_angle["text"] += "TURN LEFT"
            info_yaw_angle["color"] = (0, 0, 255)
        elif yaw_angle < config.angle_min:
            info_yaw_angle["text"] += "TURN RIGHT"
            info_yaw_angle["color"] = (0, 0, 255)
        else:
            info_yaw_angle["text"] += "ANGLE OK"
            info_yaw_angle["color"] = (0, 255, 0)

        if distance > config.distance_max:
            info_distance["text"] += "STEP FORWARD"
            info_distance["color"] = (0, 0, 255)
        elif distance < config.distance_min:
            info_distance["text"] += "STEP BACK"
            info_distance["color"] = (0, 0, 255)
        else:
            info_distance["text"] += "DISTANCE OK"
            info_distance["color"] = (0, 255, 0)

        # put_outlined_text(text, point, image)
        # print(a_h, b_h)

        # ROI領域の表示
        cv2.rectangle(
            color_image,
            (roi["left"], roi["top"]),
            (roi["right"], roi["bottom"]),
            (0, 255, 0),
            2,
        )

        # ヨー角と距離を表示する
        utils.put_outlined_text(
            info_yaw_angle["text"], (10, 20), color_image, info_yaw_angle["color"]
        )
        utils.put_outlined_text(
            info_distance["text"],
            (10, 20 + 35 * 1),
            color_image,
            info_distance["color"],
        )

        # FPSを表示する
        utils.put_outlined_text(
            "REALSENSE: {:00d} FPS".format(fps.get()), (10, 20 + 35 * 2), color_image
        )

        # 現在時刻を表示する
        utils.plot_current_time(color_image)

        # 結果ウィンドウの結合
        result_images = np.vstack((depth_h, color_image))
        """        
        images = np.hstack((
            np.vstack((np.zeros_like(color_image), depth_v)),
            np.vstack((depth_h, color_image))
            ))
        """
        # 結果を表示する
        if config.debug.imshow_enable:
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", result_images)
            cv2.waitKey(1)

        # 結果を保存する
        if config.debug.save_movie_enable:
            video_writer.write(result_images)

        # 結果をストリーミング配信
        if config.debug.streaming_enable:
            vs.streaming_image = result_images



finally:
    realsense.stop()
    realsense.join()

    if config.debug.imshow_enable:
        cv2.destroyAllWindows()

    if config.debug.save_movie_enable:
        video_writer.release()