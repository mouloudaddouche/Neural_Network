#coding:utf-8
from PySide2 import QtCore, QtGui, QtWidgets, QtGui
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor
from PySide2.QtWidgets import QMainWindow, QFileDialog, QTableView, QTabWidget, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QMessageBox, QLabel, QToolButton, QMenuBar
from copy import deepcopy
import sys
from preprocessing_text import *
from Iostream import saveColumn1Dataset,saveColumn2Dataset


class PreprocessingTextWindow(QMainWindow):
     
    def close_application(self):
        sys.exit()

    def getfile(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.selectNameFilter("Arff data files (*.arff)")
        dlg.setNameFilter("Arff data files (*.arff) ;; Csv data files (*.csv) ;; All files (*.*)")
        filenames = []
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            self.file_path = str(filenames[0]. encode('utf-8'))
            with f:
                self.textEdit.setText(filenames[0].split('/')[-1])
                self.file_name = filenames[0].split('/')[-1].split('.')[0]
        self.valider.setEnabled(True)

    def open_dataset(self):
        if '.csv' in self.file_path:
            self.df=dataset_loader_csv(self.file_path)
        else:
            self.df=dataset_loader_arff(self.file_path)

    def save(self) :
        self.dico = saveColumn1Dataset (self.file_path)
        saveColumn2Dataset (self.df, self.dico, self.file_name)

    def create_table(self):
        if self.file_path != None:
            self.open_dataset()
            self.atts = get_attributes(self.df)
            self.b=True
            for self.att in self.atts:
                if not is_nominal(self.att,self.df):
                    self.b=False
            if self.b==True:
                self.pushButton_boxplot.setEnabled(False)
            else:
                self.pushButton_boxplot.setEnabled(True)
            self.pushButton_dataset.setEnabled(True)
            self.pushButton_histo.setEnabled(True)
            self.pushButton_ok.setEnabled(True)
            self.pushButton_ok_2.setEnabled(True)
            self.tableWidget.setRowCount(get_number_instances(self.df))
            self.tableWidget.setColumnCount(get_number_attributes(self.df))
            self.tableWidget.setHorizontalHeaderLabels(get_attributes(self.df))
            self.tableWidget.setSelectionBehavior(QTableView.SelectRows)
            self.tableWidget.horizontalHeader().setStretchLastSection(True)
            self.fill_table_histogram(get_instances(self.df))
            self.missing = get_number_missing_values(self.df)
            self.get_dataset_information()

    def dataset_view(self):
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 1000, 450))
        self.frame_7.setGeometry(QtCore.QRect(10, 180, 0, 0))
        self.lbl.setGeometry(QtCore.QRect(0, 0, 0, 381))
  
    def histogram_view(self):
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 0, 450))
        self.frame_7.setGeometry(QtCore.QRect(10, 180, 1000, 450))
        self.lbl.setGeometry(QtCore.QRect(0, 0, 0, 450))
        self.combo_view() 

    def combo_view(self):
        self.lbl1.clear()
        atts=get_attributes(self.df)
        self.comboBox.clear()
        self.comboBox.addItem(" ")
        self.comboBox.addItems(atts)
        self.comboBox.activated.connect(self.display_histograms)

    


    def boxplot_view(self):
        create_tmp_folder()
        boxplot(self.df)
        self.pixmap = QtGui.QPixmap('tmp/boxplot.png')
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.frame_7.setGeometry(QtCore.QRect(10, 180, 0, 0))
        self.lbl.setPixmap(self.pixmap)
        self.lbl.setScaledContents(True)
        self.lbl.setGeometry(QtCore.QRect(0, 0, 1000, 450))
        self.lbl.show()

    def preprocess(self):
        self.copie = self.df.copy(deep=True)
        if self.file_path != None:
            if self.radioButton_simple.isChecked() == True:
                fill_missing_values(self.df)
            if self.radioButton_parclasse.isChecked() == True:
                fill_missing_values_class(self.df)
            if self.radioButton_minmax.isChecked() == True:
                min_max_normalize_dataset(self.df)
            if self.radioButton_decimale.isChecked() == True:
                decimal_normalize_dataset(self.df)           
            self.tableWidget.setRowCount(0)
            self.tableWidget.setRowCount(get_number_instances(self.df))
            i=0
            for key in list(self.df.head(0)):
                j=0
                for item in self.df[key]:
                    self.tableWidget.setItem(j, i, QTableWidgetItem(str(item)))
                    j+=1
                i+=1
            self.get_dataset_information()
            self.radioButton_simple.setChecked(False)
            self.radioButton_parclasse.setChecked(False)
            self.radioButton_minmax.setChecked(False)
            self.radioButton_decimale.setChecked(False)
            self.atts = get_attributes(self.df)
            self.b=True
            for self.att in self.atts:
                if not is_nominal(self.att,self.df):
                    self.b=False
            if (self.b== False):
                self.pushButton_boxplot.setEnabled(True)
                create_tmp_folder()
                boxplot(self.df)
                self.pixmap = QtGui.QPixmap('tmp/boxplot.png')
                self.lbl.setPixmap(self.pixmap)
                self.lbl.setScaledContents(True)
                self.lbl.show()
            self.display_histograms(self.comboBox.currentIndex())

    def undo_changes(self):
        self.df=dataset_loader_arff(self.file_path)
        self.atts = get_attributes(self.df)
        self.b=True
        for self.att in self.atts:
                if not is_nominal(self.att,self.df):
                    self.b=False
        if self.b==True:
                self.pushButton_boxplot.setEnabled(False)
        else:
                self.pushButton_boxplot.setEnabled(True)
                create_tmp_folder()
                boxplot(self.df)
                self.pixmap = QtGui.QPixmap('tmp/boxplot.png')
                self.lbl.setPixmap(self.pixmap)
                self.lbl.setScaledContents(True)
                self.lbl.show()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(get_number_instances(self.df))
        self.fill_table_histogram(get_instances(self.df))
        text = "Nombre de valeurs manquantes: " + str(self.missing)
        self.label_valmanquantes.setText(text)
        self.radioButton_simple.setChecked(False)
        self.radioButton_parclasse.setChecked(False)
        self.radioButton_minmax.setChecked(False)
        self.radioButton_decimale.setChecked(False)
        self.combo_view()

    def get_dataset_information(self):
        text="Nombre d'instances: "+str(get_number_instances(self.df))
        self.label_instances.setText(text)
        text = "Nombre d'attributs: " + str(get_number_attributes(self.df))
        self.label_attributs.setText(text)
        text = "Nombre de valeurs manquantes: " + str(get_number_missing_values(self.df))
        self.label_valmanquantes.setText(text)

    def fill_table_histogram(self, instances):
        self.comboBox.clear()
        self.combo_view()
        self.tableAttribute.setRowCount(0)
        self.label_5.setText("Type: ")
        self.label_6.setText("Valeurs manquantes: ")
        self.label_7.setText("Valeurs distinctes: ")
        i=0
        for key in instances.keys():
            j=0
            for item in instances[key]:
                self.tableWidget.setItem(i, j, QTableWidgetItem(item))
                j+=1
            i+=1

    def display_histograms(self, att):
        if(self.comboBox.currentIndex() != 0):
            create_tmp_folder()
            text="Valeurs manquantes: "+str(get_number_missing_values_att(att-1, self.df))
            self.label_6.setText(text)
            if is_nominal(self.comboBox.currentText(), self.df):
                text="Type: nominal"
                self.label_5.setText(text)
                text = "Valeurs distinctes: " + str(len(get_number_values_val_att(att-1, self.df)))
                self.label_7.setText(text)
                vals = get_number_values_val_att(att-1, self.df)
                self.tableAttribute.setRowCount( len(get_number_values_val_att(att-1, self.df)))
                self.tableAttribute.setColumnCount(2)
                self.tableAttribute.horizontalHeader().setVisible(True)
                self.tableAttribute.setHorizontalHeaderLabels(['Label','Valeur'])
                self.tableAttribute.horizontalHeader().setStretchLastSection(True)
                self.tableAttribute.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
                self.tableAttribute.verticalHeader().setVisible(False)
                i=0
                for key in vals:
                    self.tableAttribute.setItem(i, 0, QTableWidgetItem(key))
                    self.tableAttribute.setItem(i, 1, QTableWidgetItem(str(vals.get(key))))
                    i+=1
                create_tmp_folder()
                histogram_nominal(att-1, self.df)
                self.pixmap = QtGui.QPixmap('tmp/histo_'+str(att-1)+'.png')
                self.lbl1.setPixmap(self.pixmap)
                self.lbl1.setScaledContents(True)
                self.lbl1.show()
            else:
                text = "Type: numerique"
                self.label_5.setText(text)
                text = "Valeurs distinctes: " + str( len(get_number_values_val_att(att-1, self.df)))
                self.label_7.setText(text)
                self.tableAttribute.setRowCount(4)
                self.tableAttribute.setColumnCount(2)
                self.tableAttribute.horizontalHeader().setVisible(True)
                self.tableAttribute.setHorizontalHeaderLabels(['Label','Valeur'])
                self.tableAttribute.horizontalHeader().setStretchLastSection(True)
                self.tableAttribute.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
                self.tableAttribute.verticalHeader().setVisible(False)
                self.tableAttribute.setItem(0, 0, QTableWidgetItem("Minimum: "))
                self.tableAttribute.setItem(0, 1, QTableWidgetItem(str(return_min(get_att_by_index(att-1, self.df), self.df))))
                self.tableAttribute.setItem(1, 0, QTableWidgetItem("Maximum: "))
                self.tableAttribute.setItem(1, 1, QTableWidgetItem(str(return_max(get_att_by_index(att-1, self.df), self.df))))
                self.tableAttribute.setItem(2, 0, QTableWidgetItem("Moyenne: "))
                self.tableAttribute.setItem(2, 1, QTableWidgetItem(str(round(return_average(get_att_by_index(att-1, self.df), self.df),2))))
                self.tableAttribute.setItem(3, 0, QTableWidgetItem("Ecart-Type: "))
                self.tableAttribute.setItem(3, 1, QTableWidgetItem(str(round(return_stdv(get_att_by_index(att-1, self.df), self.df),2))))
                create_tmp_folder()
                histogram_numeral(att-1, self.df)
                self.pixmap = QtGui.QPixmap('tmp/histo_'+str(att-1)+'.png')
                self.lbl1.setPixmap(self.pixmap)
                self.lbl1.setScaledContents(True)
                self.lbl1.show()

    def precedent_state(self):
        self.copie_suiv = self.df.copy(deep=True)
        self.df = self.copie.copy(deep=True)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(get_number_instances(self.df))
        self.fill_table_histogram(get_instances(self.df))
        text = "Nombre de valeurs manquantes: " + str(self.missing)
        self.label_valmanquantes.setText(text)
        create_tmp_folder()
        boxplot(self.df)
        self.pixmap = QtGui.QPixmap('tmp/boxplot.png')
        self.lbl.setPixmap(self.pixmap)
        self.lbl.setScaledContents(True)
        self.lbl.show()
        self.combo_view()

    def next_state(self):
        self.copie = self.df.copy(deep=True)
        self.df = self.copie_suiv.copy(deep=True)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(get_number_instances(self.df))
        self.fill_table_histogram(get_instances(self.df))
        text = "Nombre de valeurs manquantes: " + str(self.missing)
        self.label_valmanquantes.setText(text)
        create_tmp_folder()
        boxplot(self.df)
        self.pixmap = QtGui.QPixmap('tmp/boxplot.png')
        self.lbl.setPixmap(self.pixmap)
        self.lbl.setScaledContents(True)
        self.lbl.show()
        self.combo_view()


    def __init__(self, *args, **kwargs):
        super(PreprocessingTextWindow, self).__init__(*args, **kwargs)
        self.file_path = None
        self.file_name = None
        self.copie = None
        self.copie_suiv = None
        self.missing = 0
        self.df= None
        self.setObjectName("self")
        self.resize(1280, 720)
        self.setAutoFillBackground(True)
        self.setFixedSize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setAutoFillBackground(True)
        self.tab1.setObjectName("tab1")
        self.frame = QtWidgets.QFrame(self.tab1)
        self.frame.setGeometry(QtCore.QRect(10, 60, 1000, 111))
        self.frame.setAutoFillBackground(True)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(11, 10, 184, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(100, 30, 301, 65))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_instances = QtWidgets.QLabel(self.widget)
        self.label_instances.setObjectName("label_instances")
        self.verticalLayout.addWidget(self.label_instances)
        self.label_attributs = QtWidgets.QLabel(self.widget)
        self.label_attributs.setObjectName("label_attributs")
        self.verticalLayout.addWidget(self.label_attributs)
        self.label_valmanquantes = QtWidgets.QLabel(self.widget)
        self.label_valmanquantes.setObjectName("label_valmanquantes")
        self.verticalLayout.addWidget(self.label_valmanquantes)

        self.frame_2 = QtWidgets.QFrame(self.tab1)
        self.frame_2.setGeometry(QtCore.QRect(1030, 60, 211, 261))
        self.frame_2.setAutoFillBackground(True)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.toolBox = QtWidgets.QToolBox(self.frame_2)
        self.toolBox.setGeometry(QtCore.QRect(10, 40, 191, 151))
        self.toolBox.setObjectName("toolBox")
        self.page_valmanquantes = QtWidgets.QWidget()
        self.page_valmanquantes.setGeometry(QtCore.QRect(0, 0, 191, 89))
        self.page_valmanquantes.setObjectName("page_valmanquantes")
        self.layoutWidget = QtWidgets.QWidget(self.page_valmanquantes)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 188, 52))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.radioButton_simple = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_simple.setObjectName("radioButton_simple")
        self.verticalLayout_2.addWidget(self.radioButton_simple)
        self.radioButton_parclasse = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_parclasse.setObjectName("radioButton_parclasse")
        self.verticalLayout_2.addWidget(self.radioButton_parclasse)
        self.toolBox.addItem(self.page_valmanquantes, "")
        self.page_normalisation = QtWidgets.QWidget()
        self.page_normalisation.setGeometry(QtCore.QRect(0, 0, 191, 89))
        self.page_normalisation.setObjectName("page_normalisation")
        self.layoutWidget1 = QtWidgets.QWidget(self.page_normalisation)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 0, 96, 80))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.radioButton_minmax = QtWidgets.QRadioButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setItalic(False)
        self.radioButton_minmax.setFont(font)
        self.radioButton_minmax.setObjectName("radioButton_minmax")
        self.verticalLayout_3.addWidget(self.radioButton_minmax)
        self.radioButton_decimale = QtWidgets.QRadioButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setItalic(False)
        self.radioButton_decimale.setFont(font)
        self.radioButton_decimale.setObjectName("radioButton_decimale")
        self.verticalLayout_3.addWidget(self.radioButton_decimale)
        self.toolBox.addItem(self.page_normalisation, "")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 181, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.widget1 = QtWidgets.QWidget(self.frame_2)
        self.widget1.setGeometry(QtCore.QRect(70, 220, 131, 30))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_ok = QtWidgets.QPushButton(self.widget1)
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.horizontalLayout_2.addWidget(self.pushButton_ok)
        self.pushButton_ok_2 = QtWidgets.QPushButton(self.widget1)
        self.pushButton_ok_2.setObjectName("pushButton_ok_2")
        self.horizontalLayout_2.addWidget(self.pushButton_ok_2)

        self.frame_3 = QtWidgets.QFrame(self.tab1)
        self.frame_3.setGeometry(QtCore.QRect(10, 180, 1000, 450))
        self.frame_3.setAutoFillBackground(True)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.tableWidget = QtWidgets.QTableWidget(self.frame_3)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 1000, 450))
        self.tableWidget.setAutoFillBackground(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.frame_4 = QtWidgets.QFrame(self.tab1)
        self.frame_4.setGeometry(QtCore.QRect(1030, 330, 211, 297))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
       
        self.pushButton_dataset = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_dataset.setGeometry(QtCore.QRect(49, 67, 121, 45))
        self.pushButton_dataset.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_dataset.setAutoDefault(False)
        self.pushButton_dataset.setDefault(False)
        self.pushButton_dataset.setFlat(False)
        self.pushButton_dataset.setObjectName("pushButton_dataset")
        self.pushButton_boxplot = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_boxplot.setGeometry(QtCore.QRect(50, 120, 121, 45))
        self.pushButton_boxplot.setObjectName("pushButton_boxplot")
        self.pushButton_histo = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_histo.setGeometry(QtCore.QRect(50, 170, 121, 45))
        self.pushButton_histo.setObjectName("pushButton_histo")
        self.pushButton_save = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_save.setGeometry(QtCore.QRect(50, 220, 121, 45))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 111, 17))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.layoutWidget2 = QtWidgets.QWidget(self.tab1)
        self.layoutWidget2.setGeometry(QtCore.QRect(20, 20, 988, 31))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textEdit = QtWidgets.QTextEdit(self.layoutWidget2)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.toolButton = QtWidgets.QToolButton(self.layoutWidget2)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout.addWidget(self.toolButton)
        self.valider = QtWidgets.QPushButton(self.layoutWidget2)
        self.valider.setObjectName("valider")
        self.horizontalLayout.addWidget(self.valider)
        self.tabWidget.addTab(self.tab1, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        self.setCentralWidget(self.centralwidget)
        self.lbl = QtWidgets.QLabel(self.frame_3)
        self.lbl.setGeometry(QtCore.QRect(10, 180, 0, 381))
        
        self.frame_7 = QtWidgets.QFrame(self.tab1)
        self.frame_7.setEnabled(True)
        self.frame_7.setGeometry(QtCore.QRect(10, 180, 0, 381))
        self.frame_7.setAutoFillBackground(True)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.comboBox = QtWidgets.QComboBox(self.frame_7)
        self.comboBox.setGeometry(QtCore.QRect(100, 40, 231, 31))
        self.comboBox.setObjectName("comboBox")
        self.frame_5 = QtWidgets.QFrame(self.frame_7)
        self.frame_5.setGeometry(QtCore.QRect(100, 90, 231, 351))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.tableAttribute = QtWidgets.QTableWidget(self.frame_5)
        self.tableAttribute.setGeometry(QtCore.QRect(10, 120, 211, 191))
        self.tableAttribute.setObjectName("tableAttribute")
        self.tableAttribute.setColumnCount(0)
        self.tableAttribute.setRowCount(0)
        self.widget = QtWidgets.QWidget(self.frame_5)
        self.widget.setGeometry(QtCore.QRect(11, 30, 211, 65))
        self.widget.setObjectName("widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_4.addWidget(self.label_6)
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_4.addWidget(self.label_7)
        self.frame_6 = QtWidgets.QFrame(self.frame_7)
        self.frame_6.setGeometry(QtCore.QRect(375, 40, 561, 401))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.lbl1 = QtWidgets.QLabel(self.frame_6)
        self.lbl1.setGeometry(QtCore.QRect(0, 0, 561, 401))
        self.pushButton_boxplot.setEnabled(False)
        self.pushButton_dataset.setEnabled(False)
        self.pushButton_histo.setEnabled(False)
        self.pushButton_save.setEnabled(True)
        self.valider.setEnabled(False)
        self.pushButton_ok.setEnabled(False)
        self.pushButton_ok_2.setEnabled(False)
        self.radioButton_simple.setAutoExclusive(False)
        self.radioButton_parclasse.setAutoExclusive(False)
        self.radioButton_minmax.setAutoExclusive(False)
        self.radioButton_decimale.setAutoExclusive(False)
        self.toolButton.clicked.connect(self.getfile)
        self.valider.clicked.connect(self.create_table)
        self.pushButton_ok.clicked.connect(self.preprocess)
        self.pushButton_ok_2.clicked.connect(self.undo_changes)
        self.pushButton_boxplot.clicked.connect(self.boxplot_view)
        self.pushButton_dataset.clicked.connect(self.dataset_view)
        self.pushButton_histo.clicked.connect(self.histogram_view)
        self.pushButton_save.clicked.connect(self.save)
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

        ## RETURN BUTTON
        self.frame_8 = QtWidgets.QFrame(self.tab1)
        self.frame_8.setGeometry(QtCore.QRect(1080, 10, 121, 41))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.pushButton_return = QPushButton ("Back", self.frame_8)
        self.pushButton_return.setGeometry (QtCore.QRect (0, 0, 121, 41))
        self.pushButton_return.setObjectName ("pushButton_return")

    def retranslateUi(self,MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Graphic interface"))
        self.label_2.setText(_translate("MainWindow", "Dataset information:"))
        self.label_instances.setText(_translate("MainWindow", "Instances number:"))
        self.label_attributs.setText(_translate("MainWindow", "Attributes number:"))
        self.label_valmanquantes.setText(_translate("MainWindow", "Missing values number:"))
        self.radioButton_simple.setText(_translate("MainWindow", "Simple filling"))
        self.radioButton_parclasse.setText(_translate("MainWindow", "Filling by class"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_valmanquantes), _translate("MainWindow", "Missing values"))
        self.radioButton_minmax.setText(_translate("MainWindow", "Min-Max"))
        self.radioButton_decimale.setText(_translate("MainWindow", "Decimal"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_normalisation), _translate("MainWindow", "Standardization"))
        self.label_3.setText(_translate("MainWindow", "Preprocessing tasks:"))
        self.pushButton_ok.setText(_translate("MainWindow", "Confirm"))
        self.pushButton_ok_2.setText(_translate("MainWindow", "Cancel"))
        self.pushButton_dataset.setText(_translate("MainWindow", "Data-set"))
        self.pushButton_boxplot.setText(_translate("MainWindow", "Box-plot"))
        self.pushButton_histo.setText(_translate("MainWindow", "Statistical infos"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.label_4.setText(_translate("MainWindow", "Visualization:"))
        self.label.setText(_translate("MainWindow", "Select a Dataset:"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.valider.setText(_translate("MainWindow", "Confirm"))
        self.label_5.setText(_translate("MainWindow", "Type:"))
        self.label_6.setText(_translate("MainWindow", "Missing values:"))
        self.label_7.setText(_translate("MainWindow", "Distinct values:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Preprocessing"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PreprocessingTextWindow()
    MainWindow.show()
    sys.exit(app.exec_())