#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
from __future__ import print_function
import  serial


scan_fre = 6    #雷达扫描频率 ，可调范围2~8 ，单位:hz
if __name__ == '__main__':
        listdata = []
        lastangle = 0
        ser = serial.Serial('/dev/ttyACM0', 230400)    # ubuntu，如果未修改串口别名，可通过 ll /dev 查看雷达具体端口再进行更改
        # ser = serial.Serial("COM5", 230400, timeout=5)     # window系统，需要先通过设备管理器确认串口COM号
        scan_flag = False
        while scan_flag==0:
            if scan_fre==2:
                scan_flag=ser.write([0x54,0xA2,0x04,0xD1,0x02,0x00,0x00,0x79])
            elif scan_fre==3:
                scan_flag=ser.write([0x54,0xA2,0x04,0x38,0x04,0x00,0x00,0xbb])
            elif scan_fre==4:
                scan_flag=ser.write([0x54,0xA2,0x04,0xA0,0x05,0x00,0x00,0x6f])
            elif scan_fre==5:
                scan_flag=ser.write([0x54,0xA2,0x04,0x08,0x07,0x00,0x00,0xd2])
            elif scan_fre==6:
                scan_flag=ser.write([0x54,0xA2,0x04,0x70,0x08,0x00,0x00,0xa1])
            elif scan_fre==7:
                scan_flag=ser.write([0x54,0xA2,0x04,0xD8,0x09,0x00,0x00,0x16])
            elif scan_fre==8:
                scan_flag=ser.write([0x54,0xA2,0x04,0x40,0x0B,0x00,0x00,0xc8])
