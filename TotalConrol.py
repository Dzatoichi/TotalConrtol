# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TotalControl.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import face_gen
import face_rec_ph
import face_train
import face_rec_webcam
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLineEdit


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(122, 122, 122);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Button_dataset = QtWidgets.QPushButton(self.centralwidget)
        self.Button_dataset.setGeometry(QtCore.QRect(130, 160, 200, 100))
        self.Button_dataset.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                          "font: 8pt \"MS Shell Dlg 2\";")
        self.Button_dataset.setObjectName("Button_dataset")
        self.textBox = QLineEdit(self.centralwidget)
        self.textBox.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                             "font: 8pt \"MS Shell Dlg 2\";")
        self.textBox.setGeometry(QtCore.QRect(130, 100, 201, 31))
        self.Button_trainmodel = QtWidgets.QPushButton(self.centralwidget)
        self.Button_trainmodel.setGeometry(QtCore.QRect(450, 160, 200, 100))
        self.Button_trainmodel.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                             "font: 8pt \"MS Shell Dlg 2\";")
        self.Button_trainmodel.setObjectName("Button_trainmodel")
        self.Button_webcam_recogn = QtWidgets.QPushButton(self.centralwidget)
        self.Button_webcam_recogn.setGeometry(QtCore.QRect(130, 290, 200, 100))
        self.Button_webcam_recogn.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                                "font: 5pt \"MS Shell Dlg 2\";")
        self.Button_webcam_recogn.setObjectName("Button_webcam_recogn")
        self.Button_ph_recogn = QtWidgets.QPushButton(self.centralwidget)
        self.Button_ph_recogn.setGeometry(QtCore.QRect(450, 290, 200, 100))
        self.Button_ph_recogn.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                            "font: 6pt \"MS Shell Dlg 2\";")
        self.Button_ph_recogn.setObjectName("Button_ph_recogn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Button_dataset.clicked.connect(self.face_gen)
        self.Button_trainmodel.clicked.connect(self.mass)
        self.Button_webcam_recogn.clicked.connect(face_rec_webcam.face_rec_webcam)
        self.Button_ph_recogn.clicked.connect(self.ph_rec)

    def ph_rec(self):
        fname = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg)")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Info")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setText(f"Кол-во людей = {str(face_rec_ph.face_rec_ph(fname[0]))}")
        self.Button_ph_recogn = msg.exec()


    def mass(self):
        face_train.train_model()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Info")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setText(f"Модель обучена")
        self.Button_trainmodel = msg.exec()

    def face_gen(self):
        textboxValue = self.textBox.text()
        face_gen.face_gen(textboxValue)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Info")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setText(f"Личность занесена в датасет")
        self.Button_dataset = msg.exec()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TotalConrol"))
        self.Button_dataset.setText(_translate("MainWindow", "Собрать данные"))
        self.Button_trainmodel.setText(_translate("MainWindow", "Обучить модель"))
        self.Button_webcam_recogn.setText(_translate("MainWindow", "Распознование по вебкамере"))
        self.Button_ph_recogn.setText(_translate("MainWindow", "Распознование по фото"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
