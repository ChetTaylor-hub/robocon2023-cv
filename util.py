import math
import numpy as np
from yolov5.detect import *
from serial_port import Serial_port


class Laser_ranging():
    def __init__(self, port, baudrate, bytesize=8, stopbits=1, timeout=0.8, parity='N'):
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.timeout = timeout
        self.parity = parity
        self.Laser = Serial_port(self.port,
                                 self.baudrate,
                                 self.bytesize,
                                 self.stopbits,
                                 self.timeout,
                                 self.parity)

    def Laser_ranging_on(self):
        """开启激光"""
        """由产品手册得知，往模块发送大写字母O即可开启激光"""
        self.Laser.send(send_data='0')
        data = self.Laser.receive(read_len=10)
        if data == "O,OK!\r\n":  # 开启成功返回True
            return True
        else:
            return False

    def Laser_ranging_off(self):
        """关闭激光"""
        """由产品手册得知，往模块发送大写字母C即可开启激光"""
        self.Laser.send(send_data='C')
        data = self.Laser.receive(read_len=10)
        if data == "C,OK!\r\n":  # 关闭成功返回True
            return True
        else:
            return False

    def Laser_ranging_measure(self, mode):
        """测量距离，入口参数：串口实例，测量模式"""
        """由产品手册得知，测量模式为：D->标准模式，M->慢速模式，F->快速模式"""
        """由产品手册得知，读取从串口返回数据为：'12.345m,0079'字符串类型"""
        distance = 0  # 距离定义
        ac = 0  # 精准度定义
        if mode == "standard":
            self.Laser.send(send_data='D')
            if self.Laser.com.in_waiting:
                data = self.Laser.receive(read_len=100)  # 得到返回字符串
                distance = int(data[3] + data[5:8])  # 处理得到距离，单位mm
                ac = int(data[10:-1])  # 处理得到准确度，数值越小越准确
        elif mode == "slow":
            self.Laser.send(send_data='M')
            if self.Laser.com.in_waiting:
                data = self.Laser.receive(read_len=100)  # 得到返回字符串
                distance = int(data[3] + data[5:8])  # 处理得到距离，单位mm
                ac = int(data[10:-1])  # 处理得到准确度，数值越小越准确
        elif mode == "fast":
            # self.com.write("iFACM".encode())
            if self.Laser.com.in_waiting:
                data = self.Laser.receive(read_len=100)
                # distance.append(data)
                distance = int(data[2]) * 1 + int(data[4]) * 0.1 + int(data[5]) * 0.01 + int(data[6]) * 0.001
                ac = int(data[6]) * 0.001
        return distance, ac


class Control_board():
    def __init__(self, port, baudrate, bytesize=8, stopbits=1, timeout=0.8, parity='N'):
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.timeout = timeout
        self.parity = parity
        self.mpu = Serial_port(self.port,
                               self.baudrate,
                               self.bytesize,
                               self.stopbits,
                               self.timeout,
                               self.parity)

    def send_message(self, data):
        data = 'q{}e'.format(str(data).zfill(100))
        self.mpu.send(send_data=data, )


class Pose_estimation:
    def __init__(self,
                 camera_matrix=np.array([[4161.221, 0, 1445.577], [0, 4161.221, 984.686], [0, 0, 1]]) / 4,
                 dist_coeffs=np.array([[-0.1065, 0.0793, -0.0002, -8.9263e-06, -0.0161]])):
        """相机内参：内参矩阵和畸变参数"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def P4p(self, pixel_points, world_points):
        _, a, translation_vector = cv2.solvePnP(world_points,
                                                pixel_points,
                                                self.camera_matrix,
                                                self.dist_coeffs,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        pitch = translation_vector[0] / translation_vector[2]
        yaw = translation_vector[1] / translation_vector[2]

        return pitch, yaw

    def dedistortion(self, src, mode):
        """去畸变"""
        if mode == 'image':
            dedistortion_image = cv2.undistort(src,
                                               self.camera_matrix,
                                               self.dist_coeffs,
                                               P=self.camera_matrix)
        elif mode == 'pixel':
            dedistortion_points = cv2.undistortPoints(src,
                                                      self.camera_matrix,
                                                      self.dist_coeffs,
                                                      P=self.camera_matrix)

        # undistorted_points = undistorted_points * np.array([self.inv_fx, self.inv_fy]) - np.array([self.cx_fx, self.cy_fy]) #4*2
        undistorted_points_extract = dedistortion_points[:3]  # 3*2
        undistorted_points_extract = np.insert(undistorted_points_extract, 2,
                                               np.ones(len(undistorted_points_extract)),
                                               axis=1)  # 3*3
        return dedistortion_points[0][0][0], dedistortion_points[0][0][1]

    def pixelToangle(self, x, y):
        """根据2D像素点推出旋转角度"""
        pitch = math.atan((x - self.camera_matrix[0][2]) / self.camera_matrix[0][0])
        yaw = math.atan((y - self.camera_matrix[1][2]) / self.camera_matrix[1][1])

        return pitch, yaw


class Yolov5():
    def __init__(self, weights_path, source, data, img_size, view_img, nosave):
        super(Yolov5, self).__init__()
        self.weights_path = weights_path
        self.source = source
        self.data = data
        self.img_size = img_size
        self.view_img = view_img
        self.nosave = nosave

    def run_detect(self, conn):
        self.opt = self.parse_opt(self.weights_path,
                                  self.source,
                                  self.data,
                                  self.img_size,
                                  self.view_img,
                                  self.nosave,
                                  conn)
        main(self.opt)

    def parse_opt(self, weights, source, data, img_size, view_img, nosave, conn):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / weights,
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default=source,
                            help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / data, help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=img_size,
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', type=bool, default=view_img, help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', type=bool, default=nosave, help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        parser.add_argument('--conn', default=conn, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt
