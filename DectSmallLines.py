import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# 最终检测的距离
global distance
distance = 0
# 二值化的最小阈值
global minThresh
minThresh = 28

# 霍夫直线检测
def houghStraightLines(imgContour):
    # dilate_img = cv2.dilate(imgContour, kernel=np.ones((1, 1), np.uint8))
    _, binary = cv2.threshold(imgContour, minThresh, 255, cv2.THRESH_BINARY)
    # print("houghStraightLines", minThresh)
    dilate_img = cv2.dilate(binary, kernel=np.ones((3, 3), np.uint8))
    dilate_img = cv2.erode(dilate_img, kernel=np.ones((3, 3), np.uint8))

    # cv2.imshow("dilate_img", dilate_img)

    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    newImg = np.zeros(imgContour.shape, dtype=np.uint8)
    imgContour = cv2.drawContours(newImg, contours, -1, 255, 1)

    # 霍夫变换
    minLineLength = 40 # 56
    maxLineGap = 0
    lines = cv2.HoughLines(imgContour, 1.0, np.pi / 180, minLineLength, maxLineGap)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
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
            elif (abs(x1 - 640/2) < threshold_value or abs(y1 - 480/2) < threshold_value) and \
                    (abs(x2 - 640/2) < threshold_value or abs(y2 - 480/2) < threshold_value):
                continue

            cv2.line(imgContour, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 检测出多少条线
            # number += 1
            # print(number)
            linesR.append(abs(rho))
            linesTheta.append(theta)

            xPointsList = [x1, y1]
            yPointsList = [x2, y2]
            xPoints.append(xPointsList)
            yPoints.append(yPointsList)

            # cv2.imshow("Find_lines", imgContour)
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

        elif len(linesTheta) == 2  and linesTheta[0] == linesTheta[1]:
            newLinesR.append(linesR[0])
            newLinesR.append(linesR[1])
        else:
            return -1


        # 计算两线间距离
        dist_many = []
        global distance
        if len(newLinesR) % 2 == 0:
            for j in range(int(len(newLinesR) / 2)):
                dist_one = abs(newLinesR[j * 2] - newLinesR[2 * j + 1])
                if dist_one > 5 and dist_one < 100:
                    dist_many.append(dist_one)
            if len(dist_many) > 0:
                distance = sum(dist_many) / len(dist_many)

            # 设置间距的最大值
                print("distance hough is ", distance)
                # 在图像上显示距离
                # cv2.putText(imgContour, ("min_dist=%0.2f" % distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # 绘制平行线
                for k in range(len(newXPoints)):
                    cv2.line(imgContour, (newXPoints[k][0], newXPoints[k][1]), (newYPoints[k][0], newYPoints[k][1]),
                             (0, 255, 0), 2)
                    # print(len(newXPoints))
                    # print("newXPoints :", newXPoints)
                    # cv2.imshow("image", imgContour)
                    # cv2.waitKey()

                if distance > 1 and distance < 100:
                    return distance
                else:
                    return -1
            else:
                return -1
        else:
            # print("检测位置不合理")
            return -1

def main(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # 二值化
    _, binary = cv2.threshold(gray, minThresh, 255, cv2.THRESH_BINARY)
    # print("main", minThresh)
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
    imgContour = cv2.drawContours(newImg, contours, -1,255,1)
    # cv2.imshow("imgContour", imgContour)
    # cv2.waitKey()

    # 获取图像长宽
    imgInfo = image.shape
    height = imgInfo[0]
    width = imgInfo[1]

    dilate_img = cv2.dilate(imgContour, kernel=np.ones((1, 1), np.uint8))

    # cv2.imshow("dilate_img", dilate_img)

    # 霍夫变换
    minLineLength = 80
    maxLineGap = 0
    global distance

    lines = cv2.HoughLines(dilate_img, 1.0, np.pi / 180, minLineLength, maxLineGap)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
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
            elif (abs(x1 - 640 ) < threshold_value or abs(y1 - 480) < threshold_value) and \
                    (abs(x2 - 640 ) < threshold_value or abs(y2 - 480) < threshold_value):
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
                            distance = sum(dist_many) / len(dist_many)
                            # 设置间距的最大值
                            print("distance is ", distance)
                            # 在图像上显示距离
                            if distance > 1 and distance < 100:
                                cv2.putText(imgContour, ("min_dist=%0.2f" % distance), (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # 绘制平行线
                    for k in range(len(newXPoints)):
                        cv2.line(image, (newXPoints[k][0], newXPoints[k][1]), (newYPoints[k][0], newYPoints[k][1]), (0, 255, 0), 2)
                        # print(len(newXPoints))
                        # print("newXPoints :", newXPoints)

                    break

                # -----------------------------------应该只算一次

                else:
                    dists = []
                    for i in range(2):
                        for j in range(2):
                            # print(j)
                            img_roi = dilate_img[(i * 240):(i + 1) * 240, (j * 320):((j + 1) * 320)]

                            namejpg = "Images/smalls" + str(i)+ str(j) + ".jpg"
                            cv2.imwrite(namejpg,img_roi)
                            # cv2.imshow("roi", img_roi)
                            # cv2.waitKey()
                            dist = houghStraightLines(img_roi)
                            if dist > 0:
                                dists.append(dist)

                    if sum(dists) > 0:
                        dis_final = sum(dists) / len(dists)
                        distance = dis_final
                        print("final distance 3333 is ", dis_final)
                    else:
                        print("请调整测量位置3")

                    break


        elif len(linesTheta1) == 2  and linesTheta1[0] == linesTheta1[1]:
            distance = abs(linesR[0] - linesR[1])
            print("distance 44444 is ", distance)
        else:
            dists = []
            num = 0
            for i in range(2):
                for j in range(2):
                    # print(j)
                    img_roi = dilate_img[(i * 240):(i + 1) * 240, (j * 320):((j + 1) * 320)]
                    hough_roi = img_roi.copy()

                    namejpg = "Images/smalls" + str(i) + str(j) + ".jpg"
                    cv2.imwrite(namejpg, img_roi)
                    # cv2.imshow("roi", hough_roi)
                    # cv2.waitKey()
                    dist = houghStraightLines(hough_roi)
                    if dist > 0:
                        dists.append(dist)

            if sum(dists) > 0:
                dis_final = sum(dists) / len(dists)
                print("final distance 2222 is ", dis_final)
            else:
                print("调整位置2")

    else:
        dists = []
        for i in range(2):
            for j in range(2):
                # print(j)
                img_roi = dilate_img[(i * 240):(i+1)*240, (j * 320):((j+1)* 320)]

                # namejpg = "Images/smalls" + str(i)+ str(j) + ".jpg"
                # cv2.imwrite(namejpg,img_roi)
                # cv2.imshow("roi", img_roi)
                # cv2.waitKey()
                dist = houghStraightLines(img_roi)
                if dist > 0:
                    dists.append(dist)

        if sum(dists) > 0:
            dis_final = sum(dists)/len(dists)
            print("final distance 1111 is ", dis_final)
        else:
            print("请调整检测位置1")

    distance = distance * 0.95 / 403.90345
    # cv2.putText(image, ("min_dist=%0.3f" % (distance*0.0045)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, ("min_dist=%0.3f mm" % distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('findlines', image)
    cv2.waitKey(1)

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":

    capture = cv2.VideoCapture(0)

    if capture is None:
        print("Fail to open camera")
    else:
        while (True):
            # 获取一帧
            ret, frame = capture.read()

            if frame is None:
                print("Fail to grab")
                continue
            else:
                # 将这帧转换为灰度图
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                main(frame)

            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # # # 文件夹所有图片
    # path = r"D:\BUFFER\Pycharm\Detect Line\photos"
    # for filename in os.listdir(path):  # listdir的参数是文件夹的路径
    #     filenames = path + '\\' + filename
    #     # print(filenames)
    #     img_orig = cv2.imread(filenames, 1)
    #     print(filenames)
    #
    #     if img_orig is None:
    #         print("Warning: No Pictures")
    #     else:
    #         # image = cv2.imread("Images/lines.bmp",1)
    #         # cv2.imshow("gray_lines", image)
    #         # imgContour = imageProcess(gray)
    #         main(img_orig)

    # 单张处理
    # # read the image 并做预处理
    image = cv2.imread("photos/Check_003.jpg", 1)
    # # image = cv2.imread("Images/Snap_005.jpg",1)
    configFile_xml = "wellConfig.xml"
    tree = ET.parse(configFile_xml)
    root = tree.getroot()
    secondRoot = root.find("thresholdValue")
    minThresh = int(secondRoot.text)
    print(minThresh)

    if image is None:
        print("Warning: No Pictures")
    else:
        # image = cv2.imread("Images/lines.bmp",1)
        # cv2.imshow("gray_lines", image)
        # imgContour = imageProcess(gray)
        main(image)

