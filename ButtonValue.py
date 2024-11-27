import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
# 图片路径
img = cv2.imread("photos/sample.jpg", 1)
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    pre_pt = (0, 0)
    cur_pt = (0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        # pre_pt = x,y
        # print("pre_pt", (x, y))

        thresholdValue = gray[y, x]

        configFile_xml = "config.xml"
        tree = ET.parse(configFile_xml)
        root = tree.getroot()
        secondRoot = root.find("thresholdValue")
        print(secondRoot.text)

        secondRoot.text = str(thresholdValue)
        tree.write("wellConfig.xml")

        print(thresholdValue)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=8)
        cv2.putText(img, str(thresholdValue), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", gray)
        plt.imshow(imgRGB)
        plt.show()

 # 可以放大读像素
fig = plt.figure()
def call_back(event):
    # 获取的是plt上的实际坐标
    x = int(event.xdata)
    y = int(event.ydata)

    # print(x,y)
    thresholdValue = gray[y, x]

    configFile_xml = "config.xml"
    tree = ET.parse(configFile_xml)
    root = tree.getroot()
    secondRoot = root.find("thresholdValue")
    print(secondRoot.text)

    secondRoot.text = str(thresholdValue)
    tree.write("wellConfig.xml")
    print("thresholdValue", thresholdValue)
    fig.canvas.draw_idle()  # 绘图动作实时反映在图像上

fig.canvas.mpl_connect('button_press_event', call_back)
plt.imshow(imgRGB)
plt.show()