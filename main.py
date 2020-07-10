import os
import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QMessageBox
from generate import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap


class show_img_class(Ui_MainWindow, QWidget):
    def __init__(self):
        super().__init__()
        self.picture_cv_obj = None
        self.picture2_cv_obj = None
        self.img_cv_obj = None
        self.img_list = list()
        self.click_time = time.time()
        self.open_img_str = str()
        self.open_dir_str = str()
        self.tiny_img_number = 10

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        # 双击历史图
        self.pushButton_0.clicked.connect(lambda: self.read_historical_img(0))
        self.pushButton_1.clicked.connect(lambda: self.read_historical_img(1))
        self.pushButton_2.clicked.connect(lambda: self.read_historical_img(2))
        self.pushButton_3.clicked.connect(lambda: self.read_historical_img(3))
        self.pushButton_4.clicked.connect(lambda: self.read_historical_img(4))
        self.pushButton_5.clicked.connect(lambda: self.read_historical_img(5))
        self.pushButton_6.clicked.connect(lambda: self.read_historical_img(6))
        self.pushButton_7.clicked.connect(lambda: self.read_historical_img(7))
        self.pushButton_8.clicked.connect(lambda: self.read_historical_img(8))
        self.pushButton_9.clicked.connect(lambda: self.read_historical_img(9))
        # 单机功能
        self.pushButton_10.clicked.connect(self.open_picture)  # 打开
        self.pushButton_11.clicked.connect(self.save_img)  # 保存
        self.pushButton_12.clicked.connect(self.show_picture)  # 显示原图
        self.pushButton_13.clicked.connect(self.change_img2picture)  # 置入
        self.pushButton_14.clicked.connect(self.flip_img)  # 翻转
        self.pushButton_15.clicked.connect(self.adaptiveThreshold)
        self.pushButton_16.clicked.connect(self.threshold)
        self.pushButton_17.clicked.connect(self.erode_img)
        self.pushButton_18.clicked.connect(self.dilate_img)
        self.pushButton_19.clicked.connect(self.morph_open)
        self.pushButton_20.clicked.connect(self.morph_close)
        self.pushButton_21.clicked.connect(self.morph_gradient)
        self.pushButton_22.clicked.connect(self.morph_tophat)
        self.pushButton_23.clicked.connect(self.morph_blackhat)
        self.pushButton_24.clicked.connect(lambda: self.rotate_img(0))
        self.pushButton_25.clicked.connect(self.resize_img)  # 缩放
        self.pushButton_26.clicked.connect(self.clear)
        self.pushButton_27.clicked.connect(self.bgr_equalize_hist)
        self.pushButton_28.clicked.connect(self.gray_equalize_hist)
        self.pushButton_29.clicked.connect(self.inv_threshold)
        self.pushButton_30.clicked.connect(self.show_difference)  # 差异
        self.pushButton_31.clicked.connect(self.sharpen_img)  # 锐化
        self.pushButton_32.clicked.connect(self.blur_img)  # 模糊
        self.pushButton_33.clicked.connect(self.edge_img)  # 边缘
        self.pushButton_34.clicked.connect(self.show_outline)  # 轮廓
        self.pushButton_35.clicked.connect(self.hough_circles)  # 霍夫圆
        self.pushButton_36.clicked.connect(self.hough_lines)  # 霍夫直线
        self.pushButton_37.clicked.connect(self.corner_detection)  # 角点检测
        self.pushButton_38.clicked.connect(self.template_matching)  # 模板匹配
        # 数字调节框
        self.doubleSpinBox_0.valueChanged.connect(self.threshold)
        self.doubleSpinBox_1.valueChanged.connect(self.erode_img)
        self.doubleSpinBox_2.valueChanged.connect(self.dilate_img)
        self.doubleSpinBox_3.valueChanged.connect(self.morph_open)
        self.doubleSpinBox_4.valueChanged.connect(self.morph_close)
        self.doubleSpinBox_5.valueChanged.connect(self.morph_gradient)
        self.doubleSpinBox_6.valueChanged.connect(self.morph_tophat)
        self.doubleSpinBox_7.valueChanged.connect(self.morph_blackhat)
        self.doubleSpinBox_8.valueChanged.connect(lambda: self.rotate_img(1))
        self.doubleSpinBox_9.valueChanged.connect(self.resize_img)  # 缩放
        self.doubleSpinBox_10.valueChanged.connect(self.blur_img)  # 模糊
        self.doubleSpinBox_11.valueChanged.connect(self.edge_img)  # 边沿检测
        self.doubleSpinBox_12.valueChanged.connect(self.edge_img)  # 边沿检测
        self.doubleSpinBox_13.valueChanged.connect(self.hough_circles)  # 霍夫取圆
        self.doubleSpinBox_14.valueChanged.connect(self.hough_circles)  # 霍夫取圆
        self.doubleSpinBox_15.valueChanged.connect(self.hough_circles)  # 霍夫取圆
        self.doubleSpinBox_16.valueChanged.connect(self.hough_circles)  # 霍夫取圆
        self.doubleSpinBox_17.valueChanged.connect(self.hough_lines)  # 霍夫直线
        self.doubleSpinBox_18.valueChanged.connect(self.hough_lines)  # 霍夫直线
        self.doubleSpinBox_19.valueChanged.connect(self.hough_lines)  # 霍夫直线
        self.doubleSpinBox_20.valueChanged.connect(self.corner_detection)  # 角点检测
        self.doubleSpinBox_21.valueChanged.connect(self.corner_detection)  # 角点检测
        self.doubleSpinBox_22.valueChanged.connect(self.corner_detection)  # 角点检测
        # 单选按钮
        self.radioButton.clicked.connect(lambda: self.show_layer('bgr'))
        self.radioButton_0.clicked.connect(lambda: self.show_layer('G'))
        self.radioButton_1.clicked.connect(lambda: self.show_layer('b'))
        self.radioButton_2.clicked.connect(lambda: self.show_layer('g'))
        self.radioButton_3.clicked.connect(lambda: self.show_layer('r'))
        self.radioButton_4.clicked.connect(lambda: self.show_layer('h'))
        self.radioButton_5.clicked.connect(lambda: self.show_layer('s'))
        self.radioButton_6.clicked.connect(lambda: self.show_layer('v'))
        self.radioButton_7.clicked.connect(lambda: self.show_layer('L'))
        self.radioButton_8.clicked.connect(lambda: self.show_layer('A'))
        self.radioButton_9.clicked.connect(lambda: self.show_layer('B'))
        self.radioButton_10.clicked.connect(lambda: self.show_layer('all'))
        # 下拉列表
        self.comboBox_1.currentIndexChanged.connect(self.flip_img)
        self.comboBox_2.currentIndexChanged.connect(self.adaptiveThreshold)
        self.comboBox_3.currentIndexChanged.connect(self.sharpen_img)
        self.comboBox_4.currentIndexChanged.connect(self.blur_img)
        self.comboBox_5.currentIndexChanged.connect(self.show_outline)

    def whether_open_img(funciton):
        """检查是否已打开文件的装饰器"""

        def func(self):
            if self.open_img_str:
                funciton(self)
            else:
                self.open_picture()
                funciton(self)

        return func

    def open_picture(self):
        img_path, img_type = QFileDialog.getOpenFileName(self, "打开图片", self.open_dir_str,
                                                         "All Files(*.bmp *.png *.jpg *.jpeg)")
        if not img_path:
            return
        try:
            self.open_img_str = img_path
            self.open_dir_str = os.path.dirname(self.open_img_str)
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj = cv2.imread(self.open_img_str)
            self.add_img_list(self.img_cv_obj)
            self.show_img()
            self.show_tiny_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片打开异常!', str(e))
            msg_box.exec_()

    def add_img_list(self, img):
        try:
            self.img_list.insert(0, img)
            if len(self.img_list) > self.tiny_img_number:
                self.img_list.pop()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '保存历史异常!', str(e))
            msg_box.exec_()

    def show_img(self):
        try:
            qt_img_obj = self.cv2qt(self.img_cv_obj)
            pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片显示异常!', str(e))
            msg_box.exec_()

    def show_tiny_img(self):
        try:
            for i, img in enumerate(self.img_list):
                qt_img_obj = self.cv2qt(img)
                label_name = eval('self.label_' + str(i))
                pixmap = QPixmap.fromImage(qt_img_obj).scaled(label_name.width(), label_name.height())
                label_name.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '历史图显示异常!', str(e))
            msg_box.exec_()

    def change_img2picture(self):
        try:
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj
            self.show_picture()
            self.add_img_list(self.img_cv_obj)
            self.show_tiny_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片置入异常!', str(e))
            msg_box.exec_()

    @staticmethod
    def cv2qt(img) -> QImage:
        try:
            image_height, image_width, image_depth = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, image_width, image_height, image_width * image_depth, QImage.Format_RGB888)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '类型转换异常!', str(e))
            msg_box.exec_()

    def is_double_click(self) -> bool:
        t = time.time()
        delta = t - self.click_time
        self.click_time = t
        if delta < 0.3:
            return True

    def read_historical_img(self, value):
        if not self.is_double_click():
            return
        if len(self.img_list) > value:
            self.picture_cv_obj = self.picture2_cv_obj = self.img_cv_obj = self.img_list.pop(value)
            self.add_img_list(self.img_cv_obj)
            self.show_img()
            self.show_tiny_img()

    def get_image_checkbox(self) -> cv2:
        if self.checkBox.isChecked():
            return self.img_cv_obj
        else:
            return self.picture_cv_obj

    def clear(self):
        self.label.clear()
        for i in range(len(self.img_list)):
            label_name = eval('self.label_' + str(i))
            label_name.clear()

        self.picture_cv_obj = None
        self.picture2_cv_obj = None
        self.img_cv_obj = None
        self.img_list = list()
        self.click_time = time.time()
        self.open_img_str = str()
        self.open_dir_str = str()

    @whether_open_img
    def show_picture(self):
        try:
            qt_img_obj = self.cv2qt(self.picture_cv_obj)
            pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '原图显示异常!', str(e))
            msg_box.exec_()

    @whether_open_img
    def save_img(self):
        try:
            img_name, img_type = QFileDialog.getSaveFileName(None, "保存图片", self.open_dir_str, "*.jpg;;*.png")
            if img_name:
                cv2.imwrite(img_name, self.img_cv_obj)
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '图片保存异常!', str(e))
            msg_box.exec_()

    def show_layer(self, letter):
        if not self.open_img_str:
            return self.open_picture()
        if letter == 'bgr':
            self.picture_cv_obj = self.img_cv_obj = self.picture2_cv_obj
            return self.show_img()
        bgr = self.picture2_cv_obj
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(bgr)
        h, s, v = cv2.split(hsv)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        G = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if letter == 'all':
            all = cv2.vconcat((cv2.hconcat((b, g, r)), cv2.hconcat((h, s, v)), cv2.hconcat((L, A, B))))
            all = cv2.resize(all, bgr.shape[0:2])
        img = cv2.cvtColor(eval(letter), cv2.COLOR_GRAY2BGR)
        self.picture_cv_obj = self.img_cv_obj = img
        qt_img_obj = self.cv2qt(self.img_cv_obj)
        pixmap = QPixmap.fromImage(qt_img_obj).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(pixmap)

    def rotate_img(self, value):
        """ 旋转
        :param value: click事件为0 ,change事件为1
        :return:
        """
        # 选择 & 不点击时直接返回 不操作!
        if not self.open_img_str:
            return
        if self.checkBox.isChecked() and value == 1:
            return
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_8.value())
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        # 计算旋转矩阵:(中心,角度,缩放比)
        m = cv2.getRotationMatrix2D(center, val, 1)
        # 使用openCV仿射变换实现函数旋转
        self.img_cv_obj = cv2.warpAffine(img, m, (w, h))
        self.show_img()

    @whether_open_img
    def flip_img(self):
        """翻转"""
        img = self.get_image_checkbox()
        index = self.comboBox_4.currentIndex()
        self.img_cv_obj = cv2.flip(img, index)
        self.show_img()

    @whether_open_img
    def threshold(self):
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_0.value())
        ret, self.img_cv_obj = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
        self.show_img()

    @whether_open_img
    def inv_threshold(self):
        img = self.get_image_checkbox()
        ret, self.img_cv_obj = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        self.show_img()

    @whether_open_img
    def adaptiveThreshold(self):
        img = self.get_image_checkbox()
        index = self.comboBox_1.currentIndex()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if index == 0:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY, 11, 2)
        elif index == 1:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif index == 2:
            adaptive_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # noinspection PyUnboundLocalVariable
        self.img_cv_obj = cv2.cvtColor(adaptive_img, cv2.COLOR_GRAY2BGR)
        self.show_img()

    @whether_open_img
    def erode_img(self):
        """腐蚀"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_1.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.erode(img, kernel)
        self.show_img()

    @whether_open_img
    def dilate_img(self):
        """膨胀"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_2.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.dilate(img, kernel)
        self.show_img()

    @whether_open_img
    def morph_open(self):
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_3.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        self.show_img()

    @whether_open_img
    def morph_close(self):
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_4.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        self.show_img()

    @whether_open_img
    def morph_tophat(self):
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_6.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        self.show_img()

    @whether_open_img
    def morph_blackhat(self):
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_7.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        self.show_img()

    @whether_open_img
    def morph_gradient(self):
        """形态学梯度运算"""
        img = self.get_image_checkbox()
        val = int(self.doubleSpinBox_5.value())
        kernel = np.ones((val, val), np.uint8)
        self.img_cv_obj = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        self.show_img()

    @whether_open_img
    def sharpen_img(self):
        """锐化"""
        img = self.get_image_checkbox()
        index = self.comboBox_2.currentIndex()
        # 锐化算子1
        sharpen_0 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # 锐化算子2
        sharpen_1 = np.array([[0, -1, 0], [-1, 8, -1], [0, 1, 0]]) / 4.0
        # 选择算子
        sharpen_operator = eval('sharpen_' + str(index))
        # 使用filter2D进行滤波操作
        self.img_cv_obj = cv2.filter2D(img, -1, sharpen_operator)
        self.show_img()

    @whether_open_img
    def bgr_equalize_hist(self):
        """彩色直方图均衡化"""
        img = self.get_image_checkbox()
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
        self.img_cv_obj = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.show_img()

    @whether_open_img
    def gray_equalize_hist(self):
        img = self.get_image_checkbox()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.equalizeHist(gray)
        self.img_cv_obj = cv2.cvtColor(gray_hist, cv2.COLOR_GRAY2BGR)
        self.show_img()

    @whether_open_img
    def resize_img(self):
        """缩放"""
        img = self.get_image_checkbox()
        val = self.doubleSpinBox_9.value()
        h, w = img.shape[:2]
        w, h = int(w * val), int(h * val)
        print(w, h)
        self.img_cv_obj = cv2.resize(img, (w, h))
        self.show_img()

    @whether_open_img
    def show_difference(self):
        """显示差异"""
        if len(self.img_list) < 2:
            return self.open_picture()
        a = self.img_list[0]
        b = self.img_list[1]
        self.img_cv_obj = cv2.subtract(a, b)
        self.show_img()

    @whether_open_img
    def blur_img(self):
        """模糊"""
        index = self.comboBox_3.currentIndex()
        val = int(self.doubleSpinBox_10.value())
        img = self.get_image_checkbox()
        if index == 0:
            self.img_cv_obj = cv2.medianBlur(img, val)
        elif index == 1:
            self.img_cv_obj = cv2.GaussianBlur(img, (val, val), 3)
        elif index == 2:
            self.img_cv_obj = cv2.blur(img, (val, val))
        self.show_img()

    @whether_open_img
    def edge_img(self):
        """边缘识别"""
        img = self.get_image_checkbox()
        min_val = int(self.doubleSpinBox_20.value())
        max_val = int(self.doubleSpinBox_21.value())
        gray = cv2.Canny(img, min_val, max_val)
        self.img_cv_obj = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.show_img()

    @whether_open_img
    def show_outline(self):
        """轮廓检测"""
        try:
            img = self.get_image_checkbox()
            index = self.comboBox_5.currentIndex()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 图像二值化处理，将大于阈值的设置为最大值，其它设置为0
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # 查找图像边沿：(二值化处理后的图像, 只检测外轮廓, 存储所有的轮廓点)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            img = np.zeros(img.shape, dtype=np.uint8) if self.comboBox_6.currentIndex() else img.copy()
            if index == 0:
                # 绘制边沿:(绘制图像, 轮廓点列表, 绘制全部轮廓, 轮廓颜色：红色, 轮廓粗细)
                self.img_cv_obj = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
            elif index == 1:
                x, y, w, h = cv2.boundingRect(contours[0])
                brcnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
                # 绘制边沿:(绘制图像, 轮廓点列表, 绘制全部轮廓, 轮廓颜色：红色, 轮廓粗细)
                self.img_cv_obj = cv2.drawContours(img, [brcnt], -1, (0, 0, 255), 2)
            elif index == 2:
                ellipse = cv2.fitEllipse(contours[0])  # 拟合最优椭圆
                self.img_cv_obj = cv2.ellipse(img, ellipse, (0, 0, 255), 2)  # 绘制椭圆
            elif index == 3:
                (x, y), radius = cv2.minEnclosingCircle(contours[0])
                center = (int(x), int(y))
                radius = int(radius)
                self.img_cv_obj = cv2.circle(img, center, radius, (0, 0, 255), 2)  # 绘制圆
            self.show_img()
        except Exception as e:
            msg_box = QMessageBox(QMessageBox.Warning, '未找到轮廓~!', str(e))
            msg_box.exec_()

    @whether_open_img
    def corner_detection(self):
        """Harris角点检测"""
        img = self.get_image_checkbox().copy()
        blockSize = int(self.doubleSpinBox_18.value())
        ksize = int(self.doubleSpinBox_19.value())
        k = self.doubleSpinBox_17.value()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, blockSize, ksize, k)
        # 腐蚀一下，便于标记
        dst = cv2.dilate(dst, None)
        # 角点标记为红色
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        self.img_cv_obj = img
        self.show_img()

    @whether_open_img
    def hough_lines(self):
        """霍夫直线变换"""
        img = self.get_image_checkbox().copy()
        # 二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 边沿检测
        edges = cv2.Canny(gray, 50, 150)
        point_threshold = int(self.doubleSpinBox_14.value())
        minLineLength = int(self.doubleSpinBox_15.value())
        maxLineGap = int(self.doubleSpinBox_16.value())
        # 3.统计概率霍夫线变换
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, point_threshold, minLineLength=minLineLength,
                                maxLineGap=maxLineGap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        self.img_cv_obj = img
        self.show_img()

    @whether_open_img
    def hough_circles(self):
        img = self.get_image_checkbox()
        min_val = int(self.doubleSpinBox_13.value())
        max_val = int(self.doubleSpinBox_12.value())
        blur_val = int(self.doubleSpinBox_11.value())  # 模糊值
        show_mode = self.comboBox_6.currentIndex()  # 显示方式:0 全部, 1 一个 2 两个 3 三个 4 五个
        min_dist = int(self.doubleSpinBox_22.value())  # 圆心距
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_val, blur_val), 3)
        # 霍夫圆变换
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, min_dist, minRadius=min_val, maxRadius=max_val)
        if show_mode == 0:
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心
        else:
            for index, i in enumerate(circles[0, :]):
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心
                if index + 1 == show_mode:
                    break
        self.img_cv_obj = img
        self.show_img()

    def template_matching(self):
        """模板匹配"""
        template_path, template_type = QFileDialog.getOpenFileName(self, "请选择模板~", './',
                                                                   "All Files(*.bmp *.png *.jpg *.jpeg)")
        template = cv2.imread(template_path, 0)
        h, w = template.shape[:2]
        # 相似度
        similarity = self.doubleSpinBox_23.value() / 100
        for i, img_rgb in enumerate(self.img_list):
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            print(res)
            # 取匹配程度大于%80的坐标
            loc = np.where(res >= similarity)
            for pt in zip(*loc[::-1]):  # *号表示可选参数
                bottom_right = (pt[0] + w, pt[1] + h)
                cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
            cv2.imshow(str(i), img_rgb)
        cv2.waitKey(0)


def run():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = show_img_class()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
