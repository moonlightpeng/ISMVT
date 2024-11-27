#encoding:utf-8
import os
import cv2
import threading
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PySide2.QtWidgets import *
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import *
from PySide2.QtGui import *
import numpy as np
import math
import time
import xml.dom.minidom

class Measure_Slit_Sys:
    def __init__(self):
        # load ui
        file_ui = QFile("ui/mSlit.ui")
        file_ui.open(QFile.ReadOnly)
        file_ui.close()
        self.ui = QUiLoader().load(file_ui)

        # button style
        self.ui.TakeBtn.hide()
        self.ui.TakeBtn_2.hide()
        self.ui.ValueBtn.hide()

        self.ui.DetectionBtn.setStyleSheet(
            "QPushButton#DetectionBtn {image: url(ui/resource_img/rec_icon/camera.png)} QPushButton#DetectionBtn:hover {image: url(ui/resource_img/rec_icon/camera_hv.png)}")
        self.ui.SetBtn.setStyleSheet(
            "QPushButton#SetBtn {image: url(ui/resource_img/rec_icon/set.png)} QPushButton#SetBtn:hover {image: url(ui/resource_img/rec_icon/set_hv.png)}")
        self.ui.TagBtn.setStyleSheet(
            "QPushButton#TagBtn {image: url(ui/resource_img/rec_icon/tag.png)} QPushButton#TagBtn:hover {image: url(ui/resource_img/rec_icon/tag_hv.png)}")
        # self.ui.TpBtn.setStyleSheet(
            # "QPushButton#TpBtn {image: url(ui/resource_img/rec_icon/camera.png)} QPushButton#TpBtn:hover {image: url(ui/resource_img/rec_icon/camera_hv.png)}")
        self.ui.LogoArea.setStyleSheet("QLabel#LogoArea {border-image: url(ui/resource_img/logo1.jpg)}")

        # 当前检测图片
        self.cur_measure_img_path = ''
        # camera
        self.isStop = False
        self.camera_is_open = False
        self.minThresh = 28
        self.distance = 0
        self.DPI = 0.00221
        self.error_message = QWidget()
        self.messageBox = QMessageBox()

        # open camera
        self.ui.DetectionBtn.clicked.connect(self.call_camera)
        # seting
        self.ui.SetBtn.clicked.connect(self.on_SetBtn_click)
        # 检测图片
        # self.ui.StartBtn.clicked.connect(self.start_measure)
        # self.ui.StartBtn.clicked.connect(self.set_threshold_value)
        # stopBtn test
        self.ui.EndBtn.clicked.connect(self.close_cmeara)
        # open caream
        # self.ui.TpBtn.clicked.connect(self.on_TpBtn_click)
        # take photos
        self.ui.TakeBtn.clicked.connect(self.take_photos)
        # take photos close
        self.ui.TakeBtn_2.clicked.connect(self.close_take_photos)
        # tagBtn
        self.ui.TagBtn.clicked.connect(self.on_TagBtn_click)
        #ValueBtn
        self.ui.ValueBtn.clicked.connect(self.on_ValueBtn_click)
        #
        self.take_photo = None

    def init_measure_slit_sys(self):
        # 读取单个xml文件
        dom = xml.dom.minidom.parse('wellConfig.xml')
        root = dom.documentElement
        value = root.getElementsByTagName('thresholdValue')
        self.target_value = value[0]

        if int(self.target_value.firstChild.data) == 0:
            init_widget = QWidget()
            init_message = QMessageBox()
            init_message.information(init_widget, "提示！", "请先点击[设置-拍照]完成系统初始化")

    def set_wait_stata(self):
        self.camera_is_open = True
        while self.camera_is_open:
            self.ui.ResultImg.setText("正在打开相机")
            self.ui.Display.setText(" · ·")
            time.sleep(0.7)
            self.ui.Display.setText(" · · ·")
            time.sleep(0.7)
            self.ui.Display.setText(" · · · · ")
            time.sleep(0.7)
    # 检测
    def call_camera(self):
        self.ui.TakeBtn.hide()
        self.ui.TakeBtn_2.hide()
        self.ui.label.setText("Gap spacing value")
        # 检测线程
        measuring_thread = threading.Thread(target=self.detection_slit)
        measuring_thread.start()
        # 等待相机打开线程
        measuring_wait = threading.Thread(target=self.set_wait_stata)
        measuring_wait.start()

    def detection_slit(self):
        self.capture = cv2.VideoCapture(0)
        if self.capture is None:
            self.messageBox.critical(self.error_message, "ERROR", "相机打开失败")
        else:
            self.isStop = True
            # self.ui.ResultImg.setText("开始测量")
            while (self.isStop):
                self.ui.ResultImg.setText("measuring")
                # 获取一帧
                ret, self.frame = self.capture.read()

                if self.frame is None:
                    self.ui.LenDist.setText("Fail to grab")
                    continue
                else:
                    self.main()
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                    img = QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
                    self.ui.Display.setPixmap(QPixmap.fromImage(img))
                    self.camera_is_open = False
                    # 按比例填充
                    self.ui.Display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                    self.ui.Display.setScaledContents(True)
                if cv2.waitKey(1) == ord('q'):
                    break
                    self.frame.release()
                    cv2.destroyAllWindows()
            self.isStop = False
            self.camera_is_open = False

    def close_cmeara(self):
        if self.isStop:
            self.ui.ResultImg.setText("已暂停测量,相机关闭")
            self.isStop = False

    def set_threshold_value(self):
        all_img_file = os.listdir('photos')
        file_index = len(all_img_file)
        if file_index > 0:
            last_img = all_img_file[file_index - 1]
            self.cur_measure_img_path = 'photos/' + last_img
            # 图片路径
            img = cv2.imread(self.cur_measure_img_path, 1)
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
            # 可以放大读像素
            fig = plt.figure()
            def call_back(event):

                # 获取的是plt上的实际坐标
                x = int(event.xdata)
                y = int(event.ydata)
                thresholdValue = gray[y, x]

                configFile_xml = "config.xml"
                tree = ET.parse(configFile_xml)
                root = tree.getroot()
                secondRoot = root.find("thresholdValue")

                secondRoot.text = str(thresholdValue)
                tree.write("wellConfig.xml")
                print(secondRoot.text)
                fig.canvas.draw_idle()  # 绘图动作实时反映在图像

                # if target_value.firstChild.data != secondRoot.text:
                #     widget = QWidget()
                #     init_mess = QMessageBox()
                #     init_mess.information(widget, "提示！", f"阈值设置成功：{secondRoot.text}")
                #     plt.close()

            fig.canvas.mpl_connect('button_press_event', call_back)
            plt.imshow(imgRGB)
            plt.show()
        else:
            self.messageBox.critical(self.error_message, "ERROR", "请先打开相机拍照")

    def main(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        #
        # 二值化
        _, binary = cv2.threshold(gray, self.minThresh, 255, cv2.THRESH_BINARY)
        # print("main", self.minThresh)
        # 图像取反
        # cv2.imshow("binary", binary)

        # 膨胀腐蚀操作
        dilate_img = cv2.dilate(binary, kernel=np.ones((3, 3), np.uint8))
        dilate_img = cv2.erode(dilate_img, kernel=np.ones((3, 3), np.uint8))
        # cv2.imshow("bin_img", dilate_img)
        # cv2.waitKey()

        # 轮廓
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        newImg = np.zeros(gray.shape, dtype=np.uint8)
        self.imgContour = cv2.drawContours(newImg, contours, -1, 255, 1)
        # cv2.imshow("self.imgContour", self.imgContour)
        # cv2.waitKey()

        # 获取图像长宽
        imgInfo = self.frame.shape
        height = imgInfo[0]
        width = imgInfo[1]

        dilate_img = cv2.dilate(self.imgContour, kernel=np.ones((1, 1), np.uint8))

        # cv2.imshow("dilate_img", dilate_img)

        # 霍夫变换
        minLineLength = 80
        maxLineGap = 0

        lines = cv2.HoughLines(dilate_img, 1.0, np.pi / 180, minLineLength,
                               maxLineGap)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        # print(len(lines))

        if lines is None:
            print("change positon")

        elif len(lines) >= 2:
            number = 0
            linesR = []
            linesTheta1 = []
            xPoints = []
            yPoints = []
            for line in lines:
                rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
                a = np.cos(theta)  # theta是弧度
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
                y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
                x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
                y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标

                # 删除外面的四条边线
                threshold_value = 10
                if (abs(x1) < threshold_value or abs(y1) < threshold_value) and \
                        (abs(x2) < threshold_value or abs(y2) < threshold_value):
                    continue
                elif (abs(x1 - 640) < threshold_value or abs(y1 - 480) < threshold_value) and \
                        (abs(x2 - 640) < threshold_value or abs(y2 - 480) < threshold_value):
                    continue

                # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 检测出多少条线
                # number += 1
                # print(number)
                linesR.append(abs(rho))
                linesTheta1.append(theta)

                xPointsList = [x1, y1]
                yPointsList = [x2, y2]
                xPoints.append(xPointsList)
                yPoints.append(yPointsList)

                # cv2.imshow("Find_lines", image)
                # cv2.waitKey()

            # print(linesR)
            # print(linesTheta1)
            # print(xPoints)
            newLinesR = []
            newXPoints = []
            newYPoints = []
            if len(linesTheta1) >= 3:
                # 找到相同的元素和下标 平行线
                list_same = []
                for i in linesTheta1:
                    address_index = [x for x in range(len(linesTheta1)) if linesTheta1[x] == i]
                    list_same.append([i, address_index])
                dict_address = dict(list_same)

                for values in dict_address.values():
                    # print(values)
                    if len(values) >= 2:
                        index = values
                        newLinesR.append(linesR[index[0]])
                        newLinesR.append(linesR[index[1]])

                        # 获取平行线坐标
                        newXPoints.append(xPoints[index[0]])
                        newXPoints.append(xPoints[index[1]])

                        newYPoints.append(yPoints[index[0]])
                        newYPoints.append(yPoints[index[1]])

                        # 计算两线间距离
                        dist_many = []
                        if len(newLinesR) % 2 == 0:
                            for j in range(int(len(newLinesR) / 2)):
                                dist_one = abs(newLinesR[j * 2] - newLinesR[2 * j + 1])
                                if dist_one > 5 and dist_one < 100:
                                    dist_many.append(dist_one)
                            if len(dist_many) > 0:
                                self.distance = sum(dist_many) / len(dist_many)
                                # 设置间距的最大值
                                print("self.distance is ", self.distance)
                                # 在图像上显示距离
                                if self.distance > 1 and self.distance < 100:
                                    cv2.putText(self.imgContour, ("min_dist=%0.2f" % self.distance), (50, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    self.ui.LenDist.setText("%0.3f mm" % self.distance)


                        # 绘制平行线
                        for k in range(len(newXPoints)):
                            cv2.line(self.frame, (newXPoints[k][0], newXPoints[k][1]), (newYPoints[k][0], newYPoints[k][1]),
                                     (0, 255, 0), 2)
                            # print(len(newXPoints))
                            # print("newXPoints :", newXPoints)

                        break

                    # -----------------------------------应该只算一次

                    else:
                        dists = []
                        for i in range(2):
                            for j in range(2):
                                # print(j)
                                self.img_roi = dilate_img[(i * 240):(i + 1) * 240, (j * 320):((j + 1) * 320)]

                                namejpg = "Images/smalls" + str(i) + str(j) + ".jpg"
                                cv2.imwrite(namejpg, self.img_roi)
                                # cv2.imshow("roi", self.img_roi)
                                # cv2.waitKey()
                                dist = self.houghStraightLines()
                                if dist > 0:
                                    dists.append(dist)

                        if sum(dists) > 0:
                            dis_final = sum(dists) / len(dists)
                            self.distance = dis_final
                            print("final self.distance 3333 is ", dis_final)
                        else:
                            print("请调整测量位置3")

                        break


            elif len(linesTheta1) == 2 and linesTheta1[0] == linesTheta1[1]:
                self.distance = abs(linesR[0] - linesR[1])
                print("self.distance 44444 is ", self.distance)
            else:
                dists = []
                num = 0
                for i in range(2):
                    for j in range(2):
                        # print(j)
                        self.img_roi = dilate_img[(i * 240):(i + 1) * 240, (j * 320):((j + 1) * 320)]
                        self.hough_roi = self.img_roi.copy()

                        namejpg = "Images/smalls" + str(i) + str(j) + ".jpg"
                        cv2.imwrite(namejpg, self.img_roi)
                        # cv2.imshow("roi", hough_roi)
                        # cv2.waitKey()
                        dist = self.houghStraightLines()
                        if dist > 0:
                            dists.append(dist)

                if sum(dists) > 0:
                    dis_final = sum(dists) / len(dists)
                    print("final self.distance 2222 is ", dis_final)
                else:
                    print("调整位置2")


        else:
            dists = []
            for i in range(2):
                for j in range(2):
                    # print(j)
                    self.img_roi = dilate_img[(i * 240):(i + 1) * 240, (j * 320):((j + 1) * 320)]

                    # namejpg = "Images/smalls" + str(i)+ str(j) + ".jpg"
                    # cv2.imwrite(namejpg,self.img_roi)
                    # cv2.imshow("roi", self.img_roi)
                    # cv2.waitKey()
                    dist = self.houghStraightLines()
                    if dist > 0:
                        dists.append(dist)

            if sum(dists) > 0:
                dis_final = sum(dists) / len(dists)

            else:
                print("请调整检测位置1")

        self.distance = self.distance * 0.95 / 403.90345
        # cv2.putText(image, ("min_dist=%0.3f" % (self.distance*0.0045)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(self.frame, ("min_dist=%0.3f mm" % self.distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.ui.LenDist.setText("%0.3f mm" % self.distance)
        #cv2.imshow('findlines', self.frame)
        #cv2.waitKey(1)

        #plt.imshow(self.frame)
        #plt.show()

    def houghStraightLines(self):
        # dilate_img = cv2.dilate(self.imgContour, kernel=np.ones((1, 1), np.uint8))
        _, binary = cv2.threshold(self.imgContour, self.minThresh, 255, cv2.THRESH_BINARY)
        # print("houghStraightLines", self.minThresh)
        dilate_img = cv2.dilate(binary, kernel=np.ones((3, 3), np.uint8))
        dilate_img = cv2.erode(dilate_img, kernel=np.ones((3, 3), np.uint8))

        # cv2.imshow("dilate_img", dilate_img)

        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        newImg = np.zeros(self.imgContour.shape, dtype=np.uint8)
        self.imgContour = cv2.drawContours(newImg, contours, -1, 255, 1)

        # 霍夫变换
        minLineLength = 40  # 56
        maxLineGap = 0
        lines = cv2.HoughLines(self.imgContour, 1.0, np.pi / 180, minLineLength,
                               maxLineGap)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        # print(len(lines))

        if lines is None:
            return -1
        else:
            number = 0
            linesR = []
            linesTheta = []
            xPoints = []
            yPoints = []
            for line in lines:
                rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
                a = np.cos(theta)  # theta是弧度
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
                y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
                x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
                y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标

                # x1 = int(x0 + 1000/2 * (-b))  # 直线起点横坐标
                # y1 = int(y0 + 1000/2 * (a))  # 直线起点纵坐标
                # x2 = int(x0 - 1000/2 * (-b))  # 直线终点横坐标
                # y2 = int(y0 - 1000/2 * (a))  # 直线终点纵坐标

                # 删除外面的四条边线
                threshold_value = 8
                if (abs(x1) < threshold_value or abs(y1) < threshold_value) and \
                        (abs(x2) < threshold_value or abs(y2) < threshold_value):
                    continue
                elif (abs(abs(x1) - 1000) < threshold_value or abs(abs(y1) - 1000) < threshold_value) and \
                        (abs(abs(x2) - 1000) < threshold_value or abs(abs(y2) - 1000) < threshold_value):
                    continue
                elif (abs(x1 - 640 / 2) < threshold_value or abs(y1 - 480 / 2) < threshold_value) and \
                        (abs(x2 - 640 / 2) < threshold_value or abs(y2 - 480 / 2) < threshold_value):
                    continue

                cv2.line(self.imgContour, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 检测出多少条线
                # number += 1
                # print(number)
                linesR.append(abs(rho))
                linesTheta.append(theta)

                xPointsList = [x1, y1]
                yPointsList = [x2, y2]
                xPoints.append(xPointsList)
                yPoints.append(yPointsList)

                # cv2.imshow("Find_lines", self.imgContour)
                # cv2.waitKey()

            newLinesR = []
            newXPoints = []
            newYPoints = []
            if len(linesTheta) >= 3:
                # 找到相同的元素和下标
                list_same = []
                for i in linesTheta:
                    address_index = [x for x in range(len(linesTheta)) if linesTheta[x] == i]
                    list_same.append([i, address_index])
                dict_address = dict(list_same)

                for values in dict_address.values():
                    # print(len(values))
                    if len(values) >= 2:
                        index = values
                        newLinesR.append(linesR[index[0]])
                        newLinesR.append(linesR[index[1]])

                        # 获取平行线坐标
                        newXPoints.append(xPoints[index[0]])
                        newXPoints.append(xPoints[index[1]])

                        newYPoints.append(yPoints[index[0]])
                        newYPoints.append(yPoints[index[1]])
                        break

            elif len(linesTheta) == 2 and linesTheta[0] == linesTheta[1]:
                newLinesR.append(linesR[0])
                newLinesR.append(linesR[1])
            else:
                return -1

            # 计算两线间距离
            dist_many = []
            if len(newLinesR) % 2 == 0:
                for j in range(int(len(newLinesR) / 2)):
                    dist_one = abs(newLinesR[j * 2] - newLinesR[2 * j + 1])
                    if dist_one > 5 and dist_one < 100:
                        dist_many.append(dist_one)
                if len(dist_many) > 0:
                    self.distance = sum(dist_many) / len(dist_many)

                    # 设置间距的最大值
                    print("self.distance hough is ", self.distance)
                    # 在图像上显示距离
                    # cv2.putText(self.imgContour, ("min_dist=%0.2f" % self.distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # 绘制平行线
                    for k in range(len(newXPoints)):
                        cv2.line(self.imgContour, (newXPoints[k][0], newXPoints[k][1]), (newYPoints[k][0], newYPoints[k][1]),
                                 (0, 255, 0), 2)
                        # print(len(newXPoints))
                        # print("newXPoints :", newXPoints)
                        # cv2.imshow("image", self.imgContour)
                        # cv2.waitKey()

                    if self.distance > 1 and self.distance < 100:
                        return self.distance
                    else:
                        return -1
                else:
                    return -1
            else:
                # print("检测位置不合理")
                return -1

    def pretreatment(self):
        self.frame = cv2.imread(self.cur_measure_img_path, 1)
        configFile_xml = "wellConfig.xml"
        tree = ET.parse(configFile_xml)
        root = tree.getroot()
        secondRoot = root.find("thresholdValue")
        self.minThresh = int(secondRoot.text)
        print(self.minThresh)

        if self.frame is None:
            print("Warning: No Pictures")
        else:
            # image = cv2.imread("Images/lines.bmp",1)
            # cv2.imshow("gray_lines", image)
            # imgContour = imageProcess(gray)
            # main(image)
            self.main()

    def on_SetBtn_click(self):
        self.take_photo = "set_threshold_value"
        self.camera_is_open = True
        # 打开相机,显示画面
        take_photos_thread = threading.Thread(target=self.display_camera)
        take_photos_thread.start()
        # 等待相机状态
        measuring_wait = threading.Thread(target=self.set_wait_stata)
        measuring_wait.start()
    # 标定
    def on_TagBtn_click(self):
        self.take_photo = "mainFigure"
        self.camera_is_open = True
        # 打开相机,显示画面
        take_photos_thread = threading.Thread(target=self.display_camera)
        take_photos_thread.start()
        # 等待相机状态
        measuring_wait = threading.Thread(target=self.set_wait_stata)
        measuring_wait.start()
    # 显示相机画面
    def display_camera(self):
        self.take_capture = cv2.VideoCapture(0)
        self.ui.TakeBtn.show()
        self.ui.TakeBtn_2.show()
        self.ui.ResultImg.setText("拍照")
        if self.take_capture is None:
            print("Caream Error: ")
        else:
            self.isStop = True
            while self.isStop:
                # 获取一帧
                self.camera_is_open = False
                ret, self.take_frame = self.take_capture.read()
                if self.take_frame is None:
                    print("Fail to grab")
                    self.ui.LenDist.setText("Fail to grab")
                    continue
                else:
                    self.take_frame = cv2.cvtColor(self.take_frame, cv2.COLOR_RGB2BGR)
                    img = QImage(self.take_frame.data, self.take_frame.shape[1], self.take_frame.shape[0], QImage.Format_RGB888)
                    self.ui.Display.setPixmap(QPixmap.fromImage(img))
                    # 按比例填充
                    self.ui.Display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                    self.ui.Display.setScaledContents(True)
    # 拍照
    def take_photos(self):
        self.ui.Display.setText(" ")   # 清空
        if self.take_photo == "set_threshold_value":
            self.ui.ValueBtn.show()
            self.take_frame = cv2.cvtColor(self.take_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('photos/' + "sample" + '.jpg', self.take_frame)
            self.ui.ResultImg.setText("阈值设置")
            self.isStop = False
            all_img_file = os.listdir('photos')
            file_index = len(all_img_file)

            last_img = all_img_file[file_index - 1]
            self.cur_measure_img_path = 'photos/' + last_img
            self.ui.Display.setPixmap(QPixmap.fromImage(self.cur_measure_img_path))
            # 按比例填充
            # self.ui.Display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            # self.ui.Display.setScaledContents(True)
        elif self.take_photo == "mainFigure":
            self.take_frame = cv2.cvtColor(self.take_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('circles/' + "Snap" + '.jpg', self.take_frame)
            self.ui.ResultImg.setText("DPI计算")
            self.isStop = False
            all_img_file = os.listdir('circles')
            file_index = len(all_img_file)

            last_img = all_img_file[0]
            self.cur_img_path = 'circles/' + last_img
            self.ui.Display.setPixmap(QPixmap.fromImage(self.cur_img_path))
            # 按比例填充
            # self.ui.Display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            # self.ui.Display.setScaledContents(True)

        self.ui.TakeBtn.hide()
        self.ui.TakeBtn_2.hide()

        wid = QWidget()
        info = QMessageBox()
        info.information(wid, "complete", "照片已保存")
        self.take_capture.release()

        if self.take_photo == "set_threshold_value":
            self.set_threshold_value()
        elif self.take_photo == "mainFigure":
            self.mainFigure()
    # 标定结果
    def mainFigure(self):
        w = 20
        h = 5
        img = cv2.imread("circles/Snap.jpg", 1)
        print('img: ', img)
        params = cv2.SimpleBlobDetector_Params()
        # Setup SimpleBlobDetector parameters.
        # print('params')
        # print(params)
        # print(type(params))

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10e1
        params.maxArea = 10e3
        params.minDistBetweenBlobs = 25
        # params.filterByColor = True
        params.filterByConvexity = False
        # tweak these as you see fit
        # Filter by Circularity
        # params.filterByCircularity = False
        # params.minCircularity = 0.2
        # params.blobColor = 0
        # # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87
        # Filter by Inertia
        # params.filterByInertia = True
        # params.filterByInertia = False
        # params.minInertiaRatio = 0.01

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect blobs.
        # image = cv2.resize(gray_img, (int(img.shape[1]/4),int(img.shape[0]/4)), 1, 1, cv2.INTER_LINEAR)
        # image = cv2.resize(gray_img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        minThreshValue = 120
        _, gray = cv2.threshold(gray, minThreshValue, 255, cv2.THRESH_BINARY)
        gray = cv2.resize(gray, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # plt.imshow(gray)
        # cv2.imshow("gray",gray)

        # 找到距离原点（0，0）最近和最远的点

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        # opencv
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt
        # fig = plt.figure()
        # im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        color_img = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB)

        DPIall = []

        if keypoints is not None:
            # 找到距离（0，0）最近和最远的点
            kpUpLeft = []
            disUpLeft = []
            for i in range(len(keypoints)):
                dis = math.sqrt(math.pow(keypoints[i].pt[0], 2) + math.pow(keypoints[i].pt[1], 2))
                disUpLeft.append(dis)
                kpUpLeft.append(keypoints[i].pt)

            # 找到距离（640*2，0）最近和最远的点
            kpUpRight = []
            disUpRight = []
            for i in range(len(keypoints)):
                # 最大距离坐标
                dis2 = math.sqrt(math.pow(abs(keypoints[i].pt[0] - 640 * 2), 2) + math.pow(abs(keypoints[i].pt[1]), 2))
                disUpRight.append(dis2)
                kpUpRight.append(keypoints[i].pt)

            if disUpRight and disUpLeft:
                disDownLeftIndex = disUpRight.index(max(disUpRight))
                pointDL = kpUpRight[disDownLeftIndex]

                disUpRightIndex = disUpRight.index(min(disUpRight))
                pointUR = kpUpLeft[disUpRightIndex]

                disDownRightIndex = disUpLeft.index(max(disUpLeft))
                pointDR = kpUpLeft[disDownRightIndex]

                disUpLeftIndex = disUpLeft.index(min(disUpLeft))
                pointUL = kpUpLeft[disUpLeftIndex]

                if (pointDR is not None) and (pointUL is not None) and (pointDL is not None) and (pointUR is not None):
                    # cv2.circle(color_img, (int(pointDR[0]),int(pointDR[1])), 30, (0, 255, 0),2)
                    # cv2.circle(color_img, (int(pointUL[0]),int(pointUL[1])), 30, (0, 255, 0),2)
                    # cv2.line(color_img,(int(pointDR[0]),int(pointDR[1])), (int(pointDL[0]),int(pointDL[1])),(0, 0, 255),2)
                    #
                    # cv2.circle(color_img, (int(pointDL[0]),int(pointDL[1])), 30, (0, 255, 0),2)
                    # cv2.circle(color_img, (int(pointUR[0]),int(pointUR[1])), 30, (0, 255, 0),2)
                    # cv2.line(color_img, (int(pointDL[0]),int(pointDL[1])), (int(pointUR[0]),int(pointUR[1])), (0, 0, 255), 2)
                    # cv2.line(color_img, (int(pointUL[0]),int(pointUL[1])), (int(pointUR[0]),int(pointUR[1])), (0, 0, 255), 2)

                    # 显示在原图上
                    cv2.circle(img, (int(pointDR[0] / 2), int(pointDR[1] / 2)), 10, (0, 255, 0), 2)
                    cv2.circle(img, (int(pointUL[0] / 2), int(pointUL[1] / 2)), 10, (0, 255, 0), 2)
                    cv2.line(img, (int(pointDR[0] / 2), int(pointDR[1] / 2)),
                             (int(pointUL[0] / 2), int(pointUL[1] / 2)), (0, 0, 255), 2)

                    dis_UR_DL = math.sqrt(
                        math.pow(pointUR[0] - pointDL[0], 2) + math.pow(pointUR[1] - pointDL[1], 2)) / 2
                    DPIall.append(dis_UR_DL)
                    # print(dis_UR_DL)
                    global DPI
                    # 只计算斜对角线，约束条件简单一些，增加适用性
                    DPI = (math.sqrt(2)) / sum(DPIall)

                    configFile_xml = "wellConfig.xml"
                    tree = ET.parse(configFile_xml)
                    root = tree.getroot()
                    secondRoot = root.find("DPI")
                    print(secondRoot.text)

                    secondRoot.text = str(DPI)
                    tree.write("wellConfig.xml")
                    print("DPI", DPI)

                    self.ui.label.setText("DPI")
                    self.ui.LenDist.setText(str(DPI))
                else:
                    pass
                print(DPI)

        # plt.imshow(color_img,interpolation='bicubic')
        # fname = "key points"
        # titlestr = '%s found %d keypoints' % (fname, len(keypoints))
        # plt.title(titlestr)
        # fig.canvas.set_window_title(titlestr)
        # plt.show()

        # cv2.imshow('findCorners', color_img)

        self.ui.ResultImg.setText("DPI计算完成")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.Display.setPixmap(QPixmap.fromImage(img))
        # 按比例填充
        self.ui.Display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.ui.Display.setScaledContents(True)

        # cv2.imshow('findCorners', img)
        # cv2.waitKey()

    def close_take_photos(self):
        self.isStop = False
        self.ui.ResultImg.setText("相机已关闭")
        self.take_capture.release()
        self.ui.Display.setText(" ")
        self.ui.TakeBtn.hide()
        self.ui.TakeBtn_2.hide()

    # 没用的函数
    def on_ValueBtn_click(self):
        dom = xml.dom.minidom.parse('wellConfig.xml')
        root = dom.documentElement
        value = root.getElementsByTagName('thresholdValue')
        target_value = value[0]

        if int(target_value.firstChild.data) != int(self.target_value.firstChild.data):
            init_widget = QWidget()
            init_message = QMessageBox()
            init_message.information(init_widget, "提示！", f"阈值已设置：{target_value.firstChild.data}")
            self.ui.ValueBtn.hide()

if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon("ui/resource_img/icon.png"))
    sys = Measure_Slit_Sys()
    sys.ui.show()
    sys.init_measure_slit_sys()
    app.exec_()





