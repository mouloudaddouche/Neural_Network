from PySide2 import QtCore, QtGui, QtWidgets, QtGui
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor
from PySide2.QtWidgets import QTextEdit, QMainWindow, QRadioButton, QToolBox, QFrame, QAction, QStatusBar, QMenu, QFileDialog, QTableView, QTabWidget, QWidget, QTableWidget, QTableWidgetItem, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QMessageBox, QLabel, QToolButton, QMenuBar, QLineEdit, QFormLayout
import sys
import os
from Iostream import *
from Preprocessing_img import *


class PreprocessingImageWindow(QMainWindow):

    def close_application(self):
        """
            This function close the window.
        """
        sys.exit()
    
    def save(self) :
        self.dico = saveColumn1Dataset (self.dataset_path)
        saveColumn2Dataset (self.training_data, self.dico, self.file_name)
    
    def showTable(self):
        """
            This function display the corresponding matrix to an image
        """
        self.tableWidget.setVisible (True)
        if hasattr (self, 'image') :
            self.image.setVisible (False)
        self.pushButton_table.setEnabled (False)
        self.pushButton_next.setEnabled(True)
        global index_data_pre
    
        self.frame_3.setGeometry (QtCore.QRect (10, 180, 1000, 450))
        self.tableWidget.setRowCount(int (globals.height_m))
        self.tableWidget.setColumnCount(int (globals.length_m))
        matrix_image, id_class = self.training_data[globals.index_data_pre]
        for i, row in enumerate (matrix_image) :
            for j, val in enumerate (row) :
                self.tableWidget.setItem (i, j, QTableWidgetItem (str (val)))
    
    def preprocess(self):
        """
            This function preprocess an image dataset.
        """
        if self.dataset_path != None :
            if self.input_length.text() == "0" or self.input_length.text() == "":
                self.input_length.setText("30")
    
            if self.input_height.text() == "0" or self.input_height.text() == "":
                self.input_height.setText("30")
    
            # Setting values for matrix sizes
        global length_m
        global height_m
        globals.length_m = int(self.input_length.text())
        globals.height_m = int(self.input_height.text())
        ## Preprocessing
        global input_array
        global output_array
        f = open(self.dataset_path,'r')
        categories = list()
        f.readline()
        for ligne in f.readlines() :
            var=ligne.strip().split(',')
            classe=var[1] #class
            if classe not in categories :
                categories.append(classe)
        self.training_data = Preprocessing (self.dataset_path, self.file_path)
        for i in range (len (self.training_data)) :
            matrix, id_class = self.training_data[i]
            globals.input_array.append (matrix)
            globals.output_array.append (categories[id_class])
        self.start.setEnabled (False)
        self.input_height.setEnabled(False)
        self.input_length.setEnabled (False)
        self.pushButton_table.setEnabled (True)
        self.pushButton_save.setEnabled (True)


    def open_dataset (self) :
        """
            This function looks for the csv file and initializes the Window attributes.
        """
        global dataset_name
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.selectNameFilter("Csv data files (*.csv)")
        dlg.setNameFilter("Arff data files (*.arff) ;; Csv data files (*.csv) ;; All files (*.*)")
        filenames = []
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            self.dataset_path = str(filenames[0])
            with f:
                self.textEdit.setText(filenames[0].split('/')[-1])
                self.file_name = filenames[0].split('/')[-1].split('.')[0]
        if self.dataset_path != None :
            self.file_path = os.path.dirname(self.dataset_path) + globals.directory
            globals.dataset_name = self.file_name
    
        ## RESET
        global input_height
        global input_length
        global index_data
        global index_data_pre
        global input_array
        global output_array
    
        globals.input_height = 0
        globals.input_length = 0
        globals.index_data = 0
        globals.index_data_pre = 0
        globals.input_array = []
        globals.output_array = []
        self.frame_3.setGeometry(QtCore.QRect(10, 180, 1000, 450))
        self.pushButton_next.setEnabled (False)
        self.pushButton_previous.setEnabled (False)
        self.start.setEnabled (False)
        self.valider.setEnabled(True)
        self.pushButton_save.setEnabled (False)
        self.pushButton_table.setEnabled (False)
        if hasattr (self, 'tableWidget') :
            self.tableWidget.setVisible (False)
        if hasattr (self, 'image') :
            self.image.setVisible (False)



    def get_dataset_information(self):
        """
            This function gets information about the dataset such as the number of instances and the number of classes.
        """
        text="Nombre d'instances: "+str(get_number_instances(self.dataset_path))
        self.label_instances.setText(text)
        text = "Nombre de classes: " + str (get_number_class (self.dataset_path))
        self.label_classes.setText(text)
    
        self.input_height.setEnabled(True)
        self.input_length.setEnabled(True)
        self.valider.setEnabled (False)
        self.start.setEnabled(True)
        self.pushButton_next.setEnabled (True)
        if hasattr (self, 'training_data') :
            self.pushButton_table.setEnabled (True)
    
        self.frame_3.setGeometry (QtCore.QRect (250, 180, 500, 450))
        if not hasattr (self, 'image') :
            self.image = QLabel (self.frame_3)
            self.image.setGeometry (QtCore.QRect (0,0,500, 450))
            self.image.setObjectName ("image")
        image_list = []
        global index_data
        csv_file = open (self.dataset_path, "r")
        csv_file.readline()
        for line in csv_file.readlines() :
            var = line.strip().split(',')
            image_list.append (var[0])
        csv_file.close()
        self.image.setPixmap (QtGui.QPixmap (self.file_path + image_list[globals.index_data]).scaled (500, 450))
        self.image.show()




    def previous_state(self):
        """
            This function returns to the previous instance in the dataset.
        """
        global index_data
        global index_data_pre
        if self.pushButton_table.isEnabled() or not hasattr(self, 'training_data'):
            if globals.index_data - 1 >= 0 :
                globals.index_data -= 1
            if globals.index_data - 1 < 0:
                self.pushButton_next.setEnabled(True)
                self.pushButton_previous.setEnabled(False)
    
        else :
            if globals.index_data_pre - 1 >= 0 :
                globals.index_data_pre -= 1
            if globals.index_data_pre - 1 < 0:
                self.pushButton_next.setEnabled(True)
                self.pushButton_previous.setEnabled(False)
    
        if not self.valider.isEnabled () and not hasattr(self, "training_data"):
            if hasattr (self, 'training_data') and self.start.isEnabled():
                self.pushButton_table.setEnabled(False)
            elif hasattr (self, 'training_data') :
                self.pushButton_table.setEnabled (True)
    
            self.frame_3.setGeometry (QtCore.QRect (250, 180, 500, 450))
            if not hasattr (self, 'image') :
                self.image = QLabel (self.frame_3)
                self.image.setGeometry (QtCore.QRect (0,0,500, 450))
                self.image.setObjectName ("image")
            image_list = []
            global index_data
            csv_file = open (self.dataset_path, "r")
            csv_file.readline()
            for line in csv_file.readlines() :
                var = line.strip().split(',')
                image_list.append (var[0])
            csv_file.close()
    
            self.image.setPixmap (QtGui.QPixmap (self.file_path + image_list[globals.index_data]).scaled (500, 450))
            self.image.show()
        else :
            self.showTable()


    def next_state(self):
        """
            This function goes to the next instance in the dataset.
        """
        global index_data
        global index_data_pre
        if self.pushButton_table.isEnabled() or not hasattr( self, 'training_data' ): # Image Mode
            if globals.index_data + 1 < get_number_instances (self.dataset_path):
                globals.index_data += 1
            if globals.index_data + 1 >= get_number_instances(self.dataset_path):
                self.pushButton_next.setEnabled(False)
                self.pushButton_previous.setEnabled(True)
    
        else :
            if globals.index_data_pre + 1 < get_number_instances (self.dataset_path) * 2:
                globals.index_data_pre += 1
            if globals.index_data_pre + 1 >= get_number_instances(self.dataset_path):
                self.pushButton_next.setEnabled(False)
                self.pushButton_previous.setEnabled(True)
        if not self.valider.isEnabled () and not hasattr (self, "training_data"):
            if hasattr (self, 'training_data') and self.start.isEnabled():
                self.pushButton_table.setEnabled(False)
            elif hasattr (self, 'training_data') :
                self.pushButton_table.setEnabled (True)
    
            self.frame_3.setGeometry (QtCore.QRect (250, 180, 500, 450))
            if not hasattr (self, 'image') :
                self.image = QLabel (self.frame_3)
                self.image.setGeometry (QtCore.QRect (0,0,500, 450))
                self.image.setObjectName ("image")
            image_list = []
            global index_data
            csv_file = open (self.dataset_path, "r")
            csv_file.readline()
            for line in csv_file.readlines() :
                var = line.strip().split(',')
                image_list.append (var[0])
            csv_file.close()
    
            self.image.setPixmap (QtGui.QPixmap (self.file_path + image_list[globals.index_data]).scaled (500, 450))
            self.image.show()
        else :
            self.showTable ()

## GUI
    
    def __init__(self, *args, **kwargs):
        super(PreprocessingImageWindow, self).__init__(*args, **kwargs)
        self.file_path = None
        self.file_name = None
        self.dataset_path = None


        self.setObjectName("self")
        self.resize(1280, 720)
        self.setAutoFillBackground(True)
        self.setFixedSize(1280, 720)
    
        self.centralwidget = QWidget(self)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")



        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")


        ## INFO BOX:
        self.tab1 = QWidget()
        self.tab1.setAutoFillBackground(True)
        self.tab1.setObjectName("tab1")
        self.frame = QFrame(self.tab1)
        self.frame.setGeometry(QtCore.QRect(10, 60, 1000, 111))
        self.frame.setAutoFillBackground(True)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_2 = QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(11, 10, 184, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.widget = QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(100, 30, 301, 65))
        self.widget.setObjectName("widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_instances = QLabel(self.widget)
        self.label_instances.setObjectName("label_instances")
        self.verticalLayout.addWidget(self.label_instances)
        self.label_classes = QLabel(self.widget)
        self.label_classes.setObjectName("label_classes")
        self.verticalLayout.addWidget(self.label_classes)
    
        ## Frame 2 PARAMETERS
        self.frame_2 = QFrame(self.tab1)
        self.frame_2.setGeometry(QtCore.QRect(1030, 60, 211, 261))
        self.frame_2.setAutoFillBackground(True)
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
    
        self.toolBox = QToolBox(self.frame_2)
        self.toolBox.setGeometry(QtCore.QRect(10, 40, 191, 151))
        self.toolBox.setObjectName("toolBox")
    
        self.page_resize = QWidget()
        self.page_resize.setGeometry(QtCore.QRect(100, 0, 191, 89))
        self.page_resize.setObjectName("page_resize")
    
        self.layoutWidget = QWidget(self.page_resize)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 188, 52))
        self.layoutWidget.setObjectName("layoutWidget")
    
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
    
        self.form = QFormLayout ()
    
        self.input_length = QLineEdit()
        self.input_length.setValidator (QtGui.QIntValidator())
        self.input_length.setObjectName("Choose_length")
        self.verticalLayout_2.addWidget(self.input_length)
        self.input_length.setEnabled (False)


        self.input_height = QLineEdit()
        self.input_height.setValidator (QtGui.QIntValidator())
        self.input_height.setObjectName("Choose_height")
        self.verticalLayout_2.addWidget(self.input_height)
        self.input_height.setEnabled (False)
    
        self.form.addRow ("Length :", self.input_length)
    
        self.start = QPushButton(self.frame_2)
        self.start.setObjectName("pushButton_start")
        self.verticalLayout_2.addWidget(self.start)
        self.page_resize.setLayout (self.form)
    
        self.form.addRow ("height :", self.input_height)
        self.form.addRow (self.start)


        self.input_length.setText ("40")
        self.input_height.setText ("40")



        self.toolBox.addItem(self.page_resize, "")
        self.label_3 = QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 181, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")


        ## Frame 3 DISPLAY
        self.frame_3 = QFrame(self.tab1)
        self.frame_3.setGeometry(QtCore.QRect(10, 180, 1000, 450))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
    
        ## Display matrix
        self.tableWidget = QTableWidget(self.frame_3)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 1000, 450))
        self.tableWidget.setAutoFillBackground(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)


        ## Frame 4 VISUALIZATION
        self.frame_4 = QFrame(self.tab1)
        self.frame_4.setGeometry(QtCore.QRect(1030, 330, 211, 191))
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.frame_4.setObjectName("frame_4")


        ## BUTTON FOR VISUALIZATION
        self.pushButton_previous= QPushButton(self.frame_4)
        self.pushButton_previous.setGeometry(QtCore.QRect(50, 40, 121, 41))
        self.pushButton_previous.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_previous.setAutoDefault(False)
        self.pushButton_previous.setDefault(False)
        self.pushButton_previous.setFlat(False)
        self.pushButton_previous.setObjectName("pushButton_previous")
    
        self.pushButton_next = QPushButton(self.frame_4)
        self.pushButton_next.setGeometry(QtCore.QRect(50, 85, 121, 41))
        self.pushButton_next.setObjectName("pushButton_next")


        self.pushButton_table = QPushButton (self.frame_4)
        self.pushButton_table.setGeometry (QtCore.QRect (50, 130, 121, 41))
        self.pushButton_table.setObjectName ("pushButton_table")


        ## SAVE
        self.frame_5 = QFrame(self.tab1)
        self.frame_5.setGeometry(QtCore.QRect(1080, 550, 121, 41))
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
    
        self.pushButton_save = QPushButton (self.frame_5)
        self.pushButton_save.setGeometry (QtCore.QRect (0, 0, 121, 41))
        self.pushButton_save.setObjectName ("pushButton_save")


        ## RETURN BUTTON
        self.frame_6 = QFrame(self.tab1)
        self.frame_6.setGeometry(QtCore.QRect(1080, 10, 121, 41))
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
    
        self.pushButton_return = QPushButton (self.frame_6)
        self.pushButton_return.setGeometry (QtCore.QRect (0, 0, 121, 41))
        self.pushButton_return.setObjectName ("pushButton_return")



        self.label_4 = QLabel(self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.layoutWidget2 = QWidget(self.tab1)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 20, 611, 31))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textEdit = QTextEdit(self.layoutWidget2)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.toolButton = QToolButton(self.layoutWidget2)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout.addWidget(self.toolButton)
        self.valider = QPushButton(self.layoutWidget2)
        self.valider.setObjectName("valider")
        self.horizontalLayout.addWidget(self.valider)
        self.tabWidget.addTab(self.tab1, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        self.setCentralWidget(self.centralwidget)


        self.lbl = QLabel(self.frame_3)
        self.lbl.setGeometry(QtCore.QRect(10, 180, 0, 381))


        ## ACTIVATED OR NOT
        self.tableWidget.setVisible(False)
        self.pushButton_next.setEnabled(False)
        self.pushButton_previous.setEnabled(False)
        self.valider.setEnabled(False)
        self.start.setEnabled(False)
        self.pushButton_table.setEnabled (False)
        self.pushButton_save.setEnabled (False)


        self.toolButton.clicked.connect(self.open_dataset)
        self.start.clicked.connect(self.preprocess)
        self.pushButton_previous.clicked.connect(self.previous_state)
        self.pushButton_next.clicked.connect(self.next_state)
        self.valider.clicked.connect (self.get_dataset_information)
        self.pushButton_table.clicked.connect (self.showTable)
        self.pushButton_save.clicked.connect (self.save)
    
        p = QtGui.QPalette()
        p.setColor(QtGui.QPalette.Window, QtGui.QColor(43,43,47))
        p.setColor(QtGui.QPalette.Button, QtGui.QColor(65,64,59))
        p.setColor(QtGui.QPalette.WindowText, QtGui.QColor(176,204,206))
        p.setColor(self.toolBox.backgroundRole(), QtGui.QColor(43,43,47))
        p.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255,198,107))
        self.setPalette(p)
    
        self.retranslateUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)
    
    def retranslateUi(self,MainWindow):
        _translate = QtCore.QCoreApplication.translate


        MainWindow.setWindowTitle(_translate("MainWindow", "Graphic Interface"))
        self.label_2.setText(_translate("MainWindow", "Dataset Information:"))
        self.label_instances.setText(_translate("MainWindow", "Number of instances:"))
        self.label_classes.setText(_translate("MainWindow", "Number of classes:"))
        self.label.setText(_translate("MainWindow", "Select a dataset:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Preprocessing"))


        self.toolBox.setItemText(self.toolBox.indexOf(self.page_resize), _translate("MainWindow", "Resizing"))
        self.label_3.setText(_translate("MainWindow", "Preprocessing parameters:"))
        self.start.setText(_translate("MainWindow", "Start"))


        self.pushButton_previous.setText(_translate("MainWindow", "Previous image"))
        self.pushButton_next.setText(_translate("MainWindow", "Next image"))
        self.pushButton_table.setText (_translate ("MainWindow", "Show Matrix"))
        self.pushButton_save.setText (_translate ("MainWindow", "Save"))
        self.pushButton_return.setText (_translate ("MainWindow", "Back"))




        self.label_4.setText(_translate("MainWindow", "Visualization:"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.valider.setText(_translate("MainWindow", "Validate"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PreprocessingImageWindow()
    MainWindow.show()
    sys.exit(app.exec_())