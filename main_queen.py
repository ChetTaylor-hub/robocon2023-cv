"""
一些说明：
    main_queen为程序运行主函数，通过队列来共享进程之间的数据
    main也是程序运行主函数，通过管道Pipe实现进程之间数据通信，我发现管道多进程通信有延时，还未解决

此文件包含主要内容：
    object_detect定义目标检测代码
    measuring_distance激光测距
    conveying_message_to_control取出队列数据，打包数据到主控板

日志：
    2022年11月27日
    ct
"""

import time
from util import *

# ---------------------------------------- #
# 超参数 #
# ---------------------------------------- #
hym = argparse.ArgumentParser()
hym.add_argument('--weights', default='runs/train/exp9/weights/best.pt', help='model path')
hym.add_argument('--data', default='data/pillor.yaml', help='data path')
hym.add_argument('--img_size', default=[320, 320], help='img size')
hym.add_argument('--view_img', type=bool, default=False, help='whether to view img')
hym.add_argument('--nosave', type=bool, default=False, help='whether to no save img')
hym.add_argument('--video', default='D:\study_struction\ROBOCON\\robocon-2023\score\images\\test',
                 help='video index or singe img path or img list')

"""电脑串口号：COM*，树莓派串口号：/dev/ttyAMA0"""
hym.add_argument('--laser_com_port', default='COM3', help='激光测距串口号')
hym.add_argument('--mpu_com_port', default='COM1', help='主控板串口号')
hym.add_argument('--baudrate', default=115200, help='baud rate')
opt = hym.parse_args()

# ---------------------------------------- #
# 一些全局变量 #
# ---------------------------------------- #
"""进程间通信数据"""
message = {'object_detect': None, 'laser': None}  # 用于放入队列的数据


def measuring_distance(queen):
    """激光测距函数"""

    larser = Laser_ranging(port=opt.laser_com_port,
                           baudrate=opt.baudrate)  # 激光测距功能初始化
    while not larser.Laser_ranging_on():  # 开启激光测距
        print('激光测距开启中-----')

    try:
        while True:
            time.sleep(0.1)
            distance = larser.Laser_ranging_measure(mode='fast')  # 激光测距
            message['object_detect'] = None
            message['laser'] = distance
            queen.put(message)
    except KeyboardInterrupt:
        while not larser.Laser_ranging_off():  # 关闭激光测距
            print('激光测距关闭中-----')
        larser.Laser.com_close()  # 关闭串口


def conveying_message_to_control(queen):
    """向主控板发送数据函数"""

    control_board = Control_board(port=opt.mpu_com_port,
                                  baudrate=opt.baudrate)  # 初始化与主控板的串口
    pose_estimation = Pose_estimation()
    try:
        while True:
            data = queen.get()
            if data['object_detect'] is not None:  # 取出目标检测数据
                print('目标检测{}'.format(data['object_detect']))
                data = pose_estimation.pixelToangle(data['object_detect'][0], data['object_detect'][1])
                print('旋转角度{}'.format(data))
                control_board.send_message(data['object_detect'])  # 串口发送
            if data['laser'] is not None:  # 取出激光测距数据
                print('激光测距{}'.format(data['laser']))
                control_board.send_message(data['laser'])  # 串口发送
    except KeyboardInterrupt:
        control_board.mpu.com_close()  # 关闭串口


def object_detect(queen):
    """目标检测函数"""

    cnn = Yolov5(weights_path=opt.weights,
                 source=opt.video,
                 data=opt.data,
                 img_size=opt.img_size,
                 view_img=opt.view_img,
                 nosave=opt.nosave)  # 初始化神经网络

    cnn.run_detect(queen)  # 神经网络前向推理


def run__queen(funs):
    """开启多个进程"""

    from multiprocessing import Process, Pipe, Queue, cpu_count

    # 队列实例化
    queen = Queue(maxsize=8)

    # 定义多个进程，target：函数名，args：函数参数
    process = []
    for fun in funs:
        process.append(Process(target=fun, args=(queen,)))

    assert len(process) < int(cpu_count() / 2), '进程数多余cpu核数一半，请减少进程'

    # 开启多进程
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == "__main__":
    # 多进程，通过queen队列通信
    funs = [object_detect,
            measuring_distance,
            conveying_message_to_control]
    run__queen(funs)
