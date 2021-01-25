import os
from search import search


# def __init__(self):
#     self.searchObj = search()
#     self.bigPicPos = QRect(128, 0, 768, 768)
#     self.fullScreenSize = QRect(0, 0, 1024, 768)

#-----------------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_mainFuAxzE.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def __init__(self):
        self.searchObj = search("F:\课件\创新实践\data.csv")
        self.bigPicPos = QRect(128, 0, 768, 768)
        self.fullScreenSize = QRect(0, 0, 1024, 768)

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1024, 768)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.mainFrame = QFrame(self.centralwidget)
        self.mainFrame.setObjectName(u"mainFrame")
        self.mainFrame.setGeometry(QRect(10, 10, 1001, 741))
        font = QFont()
        font.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.mainFrame.setFont(font)
        self.mainFrame.setStyleSheet(u"QFrame{\n"
"background-color: rgb(56, 58, 89);\n"
"color: rgb(220, 220, 220);\n"
"border-radius: 10px;\n"
"}")
        self.mainFrame.setFrameShape(QFrame.StyledPanel)
        self.mainFrame.setFrameShadow(QFrame.Raised)
        self.uploadButton = QPushButton(self.mainFrame)
        self.uploadButton.setObjectName(u"uploadButton")
        self.uploadButton.setGeometry(QRect(30, 380, 121, 51))
        font1 = QFont()
        font1.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font1.setPointSize(14)
        font1.setBold(False)
        font1.setWeight(50)
        self.uploadButton.setFont(font1)
        self.uploadButton.setStyleSheet(u"QPushButton{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(170, 85, 255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(237, 108, 0);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:disabled {\n"
"    background-color:rgb(98, 114, 164);\n"
"}")
        self.searchButton = QPushButton(self.mainFrame)
        self.searchButton.setObjectName(u"searchButton")
        self.searchButton.setGeometry(QRect(160, 380, 121, 51))
        self.searchButton.setFont(font1)
        self.searchButton.setStyleSheet(u"QPushButton{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(170, 85, 255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(237, 108, 0);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:disabled {\n"
"    background-color:rgb(98, 114, 164);\n"
"}")
        self.uploadLabel = QLabel(self.mainFrame)
        self.uploadLabel.setObjectName(u"uploadLabel")
        self.uploadLabel.setGeometry(QRect(30, 30, 251, 251))
        self.uploadLabel.setStyleSheet(u"QLabel{\n"
"    shadow: 0 8px 16px 0 rgba(255,255,255,0.2), 0 6px 20px 0 rgba(255,255,255,0.19);\n"
"}")
        self.uploadLabel.setPixmap(QPixmap(u"ui_pics/to_be_upload.jpg"))
        self.uploadLabel.setScaledContents(True)
        self.descriptions = QLabel(self.mainFrame)
        self.descriptions.setObjectName(u"descriptions")
        self.descriptions.setGeometry(QRect(30, 450, 311, 261))
        self.descriptions.setFont(font1)
        self.descriptions.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 170, 255)\n"
"}")
        self.descriptions.setScaledContents(False)
        self.resultLabel_1 = QLabel(self.mainFrame)
        self.resultLabel_1.setObjectName(u"resultLabel_1")
        self.resultLabel_1.setGeometry(QRect(370, 30, 211, 211))
        self.resultLabel_1.setPixmap(QPixmap(u"ui_pics/1.jpg"))
        self.resultLabel_1.setScaledContents(True)
        self.resultLabel_3 = QLabel(self.mainFrame)
        self.resultLabel_3.setObjectName(u"resultLabel_3")
        self.resultLabel_3.setGeometry(QRect(370, 260, 211, 211))
        self.resultLabel_3.setPixmap(QPixmap(u"ui_pics/2.jpg"))
        self.resultLabel_3.setScaledContents(True)
        self.resultLabel_5 = QLabel(self.mainFrame)
        self.resultLabel_5.setObjectName(u"resultLabel_5")
        self.resultLabel_5.setGeometry(QRect(370, 490, 211, 211))
        self.resultLabel_5.setPixmap(QPixmap(u"ui_pics/3.jpg"))
        self.resultLabel_5.setScaledContents(True)
        self.resultLabel_4 = QLabel(self.mainFrame)
        self.resultLabel_4.setObjectName(u"resultLabel_4")
        self.resultLabel_4.setGeometry(QRect(600, 260, 211, 211))
        self.resultLabel_4.setPixmap(QPixmap(u"ui_pics/4.jpg"))
        self.resultLabel_4.setScaledContents(True)
        self.resultLabel_6 = QLabel(self.mainFrame)
        self.resultLabel_6.setObjectName(u"resultLabel_6")
        self.resultLabel_6.setGeometry(QRect(600, 490, 211, 211))
        self.resultLabel_6.setPixmap(QPixmap(u"ui_pics/5.jpg"))
        self.resultLabel_6.setScaledContents(True)
        self.resultLabel_2 = QLabel(self.mainFrame)
        self.resultLabel_2.setObjectName(u"resultLabel_2")
        self.resultLabel_2.setEnabled(True)
        self.resultLabel_2.setGeometry(QRect(600, 30, 211, 211))
        self.resultLabel_2.setPixmap(QPixmap(u"ui_pics/6.jpg"))
        self.resultLabel_2.setScaledContents(True)
        self.lastPageButton = QPushButton(self.mainFrame)
        self.lastPageButton.setObjectName(u"lastPageButton")
        self.lastPageButton.setGeometry(QRect(850, 260, 121, 61))
        font2 = QFont()
        font2.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font2.setPointSize(16)
        font2.setBold(False)
        font2.setWeight(50)
        self.lastPageButton.setFont(font2)
        self.lastPageButton.setStyleSheet(u"QPushButton{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(170, 85, 255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(237, 108, 0);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:disabled {\n"
"    background-color:rgb(98, 114, 164);\n"
"}")
        self.nextPageButton = QPushButton(self.mainFrame)
        self.nextPageButton.setObjectName(u"nextPageButton")
        self.nextPageButton.setEnabled(True)
        self.nextPageButton.setGeometry(QRect(850, 420, 121, 61))
        self.nextPageButton.setFont(font2)
        self.nextPageButton.setStyleSheet(u"QPushButton{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(170, 85, 255);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(237, 108, 0);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:disabled {\n"
"    background-color:rgb(98, 114, 164);\n"
"}")
        self.pageNumber = QLabel(self.mainFrame)
        self.pageNumber.setObjectName(u"pageNumber")
        self.pageNumber.setGeometry(QRect(850, 350, 111, 41))
        font3 = QFont()
        font3.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font3.setPointSize(12)
        font3.setBold(True)
        font3.setWeight(75)
        self.pageNumber.setFont(font3)
        self.pageNumber.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 170, 255)\n"
"}")
        self.pageNumber.setScaledContents(False)
        self.pageNumber.setAlignment(Qt.AlignCenter)
        self.pictureLink_1 = QPushButton(self.mainFrame)
        self.pictureLink_1.setObjectName(u"pictureLink_1")
        self.pictureLink_1.setGeometry(QRect(370, 30, 211, 211))
        self.pictureLink_1.setFlat(True)
        self.pictureLink_2 = QPushButton(self.mainFrame)
        self.pictureLink_2.setObjectName(u"pictureLink_2")
        self.pictureLink_2.setGeometry(QRect(600, 30, 211, 211))
        self.pictureLink_2.setFlat(True)
        self.pictureLink_3 = QPushButton(self.mainFrame)
        self.pictureLink_3.setObjectName(u"pictureLink_3")
        self.pictureLink_3.setGeometry(QRect(370, 260, 211, 211))
        self.pictureLink_3.setFlat(True)
        self.pictureLink_4 = QPushButton(self.mainFrame)
        self.pictureLink_4.setObjectName(u"pictureLink_4")
        self.pictureLink_4.setGeometry(QRect(600, 260, 211, 211))
        self.pictureLink_4.setFlat(True)
        self.pictureLink_5 = QPushButton(self.mainFrame)
        self.pictureLink_5.setObjectName(u"pictureLink_5")
        self.pictureLink_5.setGeometry(QRect(370, 490, 211, 211))
        self.pictureLink_5.setFlat(True)
        self.pictureLink_6 = QPushButton(self.mainFrame)
        self.pictureLink_6.setObjectName(u"pictureLink_6")
        self.pictureLink_6.setGeometry(QRect(600, 490, 211, 211))
        self.pictureLink_6.setFlat(True)
        self.quitButton = QPushButton(self.mainFrame)
        self.quitButton.setObjectName(u"quitButton")
        self.quitButton.setGeometry(QRect(960, 0, 41, 41))
        font4 = QFont()
        font4.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        font4.setPointSize(12)
        font4.setBold(False)
        font4.setWeight(50)
        self.quitButton.setFont(font4)
        self.quitButton.setStyleSheet(u"QPushButton{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(98, 114, 164);\n"
"	border-radius: 10px;\n"
"}\n"
"QPushButton:hover{\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(237, 108, 0);\n"
"	border-radius: 10px;\n"
"}")
        self.creditLabel = QLabel(self.mainFrame)
        self.creditLabel.setObjectName(u"creditLabel")
        self.creditLabel.setGeometry(QRect(890, 700, 111, 41))
        self.creditLabel.setFont(font3)
        self.creditLabel.setStyleSheet(u"QLabel{\n"
"	color: rgb(98, 114, 164)\n"
"}")
        self.creditLabel.setScaledContents(False)
        self.creditLabel.setAlignment(Qt.AlignCenter)
        self.pathLabel = QLabel(self.mainFrame)
        self.pathLabel.setObjectName(u"pathLabel")
        self.pathLabel.setGeometry(QRect(30, 290, 251, 21))
        font5 = QFont()
        font5.setFamily(u"\u5fae\u8f6f\u96c5\u9ed1")
        self.pathLabel.setFont(font5)
        self.pageSettingLabel = QLabel(self.mainFrame)
        self.pageSettingLabel.setObjectName(u"pageSettingLabel")
        self.pageSettingLabel.setGeometry(QRect(50, 330, 151, 31))
        self.pageSettingLabel.setFont(font1)
        self.pageSettingLabel.setStyleSheet(u"QLabel{\n"
"	color: rgb(255, 170, 255)\n"
"}")
        self.pageSettingLabel.setScaledContents(False)
        self.pageSpinBox = QSpinBox(self.mainFrame)
        self.pageSpinBox.setObjectName(u"pageSpinBox")
        self.pageSpinBox.setGeometry(QRect(210, 330, 41, 31))
        self.pageSpinBox.setFont(font)
        self.pageSpinBox.setMinimum(1)
        self.pageSpinBox.setMaximum(99)
        self.pageSpinBox.setValue(5)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.uploadButton.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u56fe\u7247", None))
        self.searchButton.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u7d22", None))
        self.uploadLabel.setText("")
        self.descriptions.setText(QCoreApplication.translate("MainWindow", u"\u5e73\u53f0\u4f7f\u7528\u8bf4\u660e\uff1a\n"
"\n"
"  1.\u70b9\u51fb\u4e0a\u4f20\u56fe\u7247\u6309\u952e\n"
"\n"
"  2.\u5728\u5f39\u51fa\u7684\u6d4f\u89c8\u6846\u4e2d\u6253\u5f00\n"
"    \u9700\u8981\u68c0\u7d22\u7684\u56fe\u7247\n"
"\n"
"  3.\u70b9\u51fb\u68c0\u7d22\u8fdb\u884c\u68c0\u7d22", None))
        self.resultLabel_1.setText("")
        self.resultLabel_3.setText("")
        self.resultLabel_5.setText("")
        self.resultLabel_4.setText("")
        self.resultLabel_6.setText("")
        self.resultLabel_2.setText("")
        self.lastPageButton.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u9875", None))
        self.nextPageButton.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u9875", None))
        self.pageNumber.setText("")
        self.pictureLink_1.setText("")
        self.pictureLink_2.setText("")
        self.pictureLink_3.setText("")
        self.pictureLink_4.setText("")
        self.pictureLink_5.setText("")
        self.pictureLink_6.setText("")
        self.quitButton.setText(QCoreApplication.translate("MainWindow", u"X", None))
        self.creditLabel.setText(QCoreApplication.translate("MainWindow", u"SUSTech", None))
        self.pathLabel.setText(QCoreApplication.translate("MainWindow", u"C:/Users/Administrator/Desktop", None))
        self.pageSettingLabel.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\u68c0\u7d22\u9875\u6570\uff1a", None))
    # retranslateUi

#-----------------------------------------------------------------------------------------------------------------------
        self.uploadButton.clicked.connect(self.upload)
        self.quitButton.clicked.connect(self.quit)
        self.searchButton.clicked.connect(self.search)
        self.nextPageButton.clicked.connect(self.nextPage)
        self.lastPageButton.clicked.connect(self.lastPage)
        self.searchButton.setDisabled(True)
        self.pictureLink_1.setDisabled(True)
        self.pictureLink_2.setDisabled(True)
        self.pictureLink_3.setDisabled(True)
        self.pictureLink_4.setDisabled(True)
        self.pictureLink_5.setDisabled(True)
        self.pictureLink_6.setDisabled(True)
        self.nextPageButton.setDisabled(True)
        self.lastPageButton.setDisabled(True)
        self.pictureLink_1.clicked.connect(self.link1)
        self.pictureLink_2.clicked.connect(self.link2)
        self.pictureLink_3.clicked.connect(self.link3)
        self.pictureLink_4.clicked.connect(self.link4)
        self.pictureLink_5.clicked.connect(self.link5)
        self.pictureLink_6.clicked.connect(self.link6)

    def zoomIn(self, label, button):
        label.setGeometry(self.bigPicPos)
        button.setGeometry(self.fullScreenSize)
        label.raise_()
        button.raise_()

    def zoomOut(self, label, button):
        label.setGeometry(self.saveGeometry)
        button.setGeometry(self.saveGeometry)

    def link1(self):
        if self.pictureLink_1.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_1.geometry()
            self.zoomIn(self.resultLabel_1, self.pictureLink_1)
        else:
            self.zoomOut(self.resultLabel_1, self.pictureLink_1)

    def link2(self):
        if self.pictureLink_2.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_2.geometry()
            self.zoomIn(self.resultLabel_2, self.pictureLink_2)
        else:
            self.zoomOut(self.resultLabel_2, self.pictureLink_2)

    def link3(self):
        if self.pictureLink_3.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_3.geometry()
            self.zoomIn(self.resultLabel_3, self.pictureLink_3)
        else:
            self.zoomOut(self.resultLabel_3, self.pictureLink_3)

    def link4(self):
        if self.pictureLink_4.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_4.geometry()
            self.zoomIn(self.resultLabel_4, self.pictureLink_4)
        else:
            self.zoomOut(self.resultLabel_4, self.pictureLink_4)

    def link5(self):
        if self.pictureLink_5.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_5.geometry()
            self.zoomIn(self.resultLabel_5, self.pictureLink_5)
        else:
            self.zoomOut(self.resultLabel_5, self.pictureLink_5)

    def link6(self):
        if self.pictureLink_6.geometry() != self.fullScreenSize:
            self.saveGeometry = self.resultLabel_6.geometry()
            self.zoomIn(self.resultLabel_6, self.pictureLink_6)
        else:
            self.zoomOut(self.resultLabel_6, self.pictureLink_6)

    def nextPage(self):
        self.turnPage(True)

    def lastPage(self):
        self.turnPage(False)

    def turnPage(self, dir):
        if dir:
            if self.currentPageNum < self.totalPageNum:
                self.currentPageNum += 1
        else:
            if self.currentPageNum > 1:
                self.currentPageNum -= 1
        self.resultLabel_1.setPixmap(QPixmap(self.resultPaths[0 + (self.currentPageNum - 1) * 6]))
        self.resultLabel_2.setPixmap(QPixmap(self.resultPaths[1 + (self.currentPageNum - 1) * 6]))
        self.resultLabel_3.setPixmap(QPixmap(self.resultPaths[2 + (self.currentPageNum - 1) * 6]))
        self.resultLabel_4.setPixmap(QPixmap(self.resultPaths[3 + (self.currentPageNum - 1) * 6]))
        self.resultLabel_5.setPixmap(QPixmap(self.resultPaths[4 + (self.currentPageNum - 1) * 6]))
        self.resultLabel_6.setPixmap(QPixmap(self.resultPaths[5 + (self.currentPageNum - 1) * 6]))
        self.pageNumber.setText(str(self.currentPageNum) + ' / ' + str(self.totalPageNum))

    def search(self):
        self.resultPaths = self.searchObj.search(self.pageSpinBox.value() * 6, self.targetPath)
        self.resultLabel_1.setPixmap(QPixmap(self.resultPaths[0]))
        self.resultLabel_2.setPixmap(QPixmap(self.resultPaths[1]))
        self.resultLabel_3.setPixmap(QPixmap(self.resultPaths[2]))
        self.resultLabel_4.setPixmap(QPixmap(self.resultPaths[3]))
        self.resultLabel_5.setPixmap(QPixmap(self.resultPaths[4]))
        self.resultLabel_6.setPixmap(QPixmap(self.resultPaths[5]))
        self.currentPageNum = 1
        self.totalPageNum = self.pageSpinBox.value()
        self.pageNumber.setText(str(self.currentPageNum) + ' / ' + str(self.totalPageNum))
        self.pictureLink_1.setEnabled(True)
        self.pictureLink_2.setEnabled(True)
        self.pictureLink_3.setEnabled(True)
        self.pictureLink_4.setEnabled(True)
        self.pictureLink_5.setEnabled(True)
        self.pictureLink_6.setEnabled(True)
        self.nextPageButton.setEnabled(True)
        self.lastPageButton.setEnabled(True)


    def upload(self):
        openfile_name = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg)")
        if openfile_name[0]:
            if len(openfile_name[0]) < 33:
                self.setText(self.pathLabel, openfile_name[0])
            else:
                self.setText(self.pathLabel, openfile_name[0][:30] + '...')
            self.uploadLabel.setPixmap(QPixmap(openfile_name[0]))
            self.targetPath = openfile_name[0]
            self.searchButton.setEnabled(True)

    def quit(self):
        os._exit(0)

    @staticmethod
    def setText(label, text):
            label.setText(text)
            label.adjustSize()