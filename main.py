from util import *
import cv2
from yolov5.detect import center_x, center_y
import _thread as thread
from multiprocessing import Process

'''相机参数'''
video = 1  # 视频来源，usb摄像头一般为0
'''串口号'''
laser_com_port = "COM10"  # 树莓派串口号，字符串格式/dev/ttyAMA0
mpu_com_port = "COM14"
baudrate = 115200

"""不可更改变量"""
message = {'object_detect': None,
           'laser': None}


def measuring_distance(conn):
    larser = Laser_ranging(port=laser_com_port,
                           baudrate=baudrate)
    # larser.Laser_ranging_on()
    try:
        while True:
            message['laser'] = None
            distance = larser.Laser_ranging_measure(mode='fast')
            message['laser'] = distance
            conn.send(message)
    except KeyboardInterrupt:
        larser.Laser.com_close()


def conveying_message_to_control(queen):
    control_board = Control_board(port=mpu_com_port,
                                  baudrate=baudrate)
    try:
        while True:
            data = queen.get()
            if data['object_detect'] is not None:
                print('目标检测{}'.format(data['object_detect']))
                # control_board.send_message(data['object_detect'])
            if data['laser'] is not None:
                pass
                # print('激光测距{}'.format(data['laser']))
                # control_board.send_message(data['laser'])
    except KeyboardInterrupt:
        control_board.com.com_close()


def object_detect(conn):
    cnn = Yolov5(weights_path='yolov5n.pt',
                 source=0,
                 data='data/coco128.yaml',
                 img_size=[320, 320],
                 view_img=True,
                 nosave=True)

    cnn.run_detect(conn)


def pri(conn):
    while True:
        center_x, center_y = conn.recv()
        for x, y in zip(center_x, center_y):
            print('进程2：长度：%d, x-->%d, y-->%d' % (len(center_x), x, y))


def run__pipe():
    from multiprocessing import Process, Pipe, cpu_count

    # 管道只有两端，可以互发数据
    conn1, conn2 = Pipe()
    a = cpu_count()
    # 定义多个进程，target：函数名，args：函数参数
    process = [Process(target=object_detect, args=(conn1,)),
               Process(target=measuring_distance, args=(conn1,)),
               Process(target=conveying_message_to_control, args=(conn2,))]

    assert len(process) < int(cpu_count() / 2), '进程数多余cpu核数一半，请减少进程'

    [p.start() for p in process]
    '''
    print('| Main', 'send')
    conn1.send(None)
    print('| Main', conn2.recv())
    '''
    [p.join() for p in process]


if __name__ == "__main__":
    # 多进程，通过Pipe管道通信
    run__pipe()
