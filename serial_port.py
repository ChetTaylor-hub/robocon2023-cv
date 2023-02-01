import serial


class Serial_port(serial.Serial):
    def __init__(self, port, baudrate, bytesize, stopbits, timeout, parity, **kwargs):
        super(Serial_port, self).__init__()
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.timeout = timeout
        self.parity = parity
        """建立串口连接"""
        """设置串口参数：串口号，波特率，数据长度，停止位,延迟时间，校验位"""
        self.com = self.com_connect()

    def com_connect(self):
        isconnect = False
        while isconnect is False:
            mpu_ser = serial.Serial(port=self.port,
                                    baudrate=self.baudrate,
                                    bytesize=self.bytesize,
                                    stopbits=self.stopbits,
                                    timeout=self.timeout,
                                    parity=self.parity)
            isconnect = mpu_ser.is_open
            if isconnect:
                print("%s mpu connection success\r\n" % self.port)
                return mpu_ser

    def com_close(self):
        """关闭串口"""
        self.com.close()

    def send(self, send_data):
        """封装好的串口发送"""
        send_len = self.com.write(str(send_data).encode())
        return send_len

    def receive(self, read_len):
        """封装好的串口接收"""
        data = self.com.read(read_len).decode('utf-8')
        return data
