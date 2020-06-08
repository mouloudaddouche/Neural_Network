import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog,QMessageBox
from PySide2.QtCore import Qt, QFile, QSize, QObject, Signal, Slot
from PySide2.QtGui import QPalette, QPixmap, QFont, QFontDatabase, QIcon
from network import Network
from fullyConnected_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from activations import *
from losses import *
from convolutional_layer import ConvLayer
from flatten_layer import FlattenLayer
from Iostream import neural_network_loader, saveColumn1Dataset,pickle

class TestingWindow(QWidget) :

    # Boolean indicating if the network is correctly loaded 
    __networkLoaded = False
    # Boolean indicating if the test file is correctly loaded
    __testFileLoaded = False

    # Network
    __network = None

    # dico 
    __dico=None

    # Load network weights
    @Slot()
    def loadNetwork(self) :
        weightsFilepath = QFileDialog.getOpenFileName(self, "Neural network weights loading", "/home", "HDF5 Files (*.h5)")[0]
        modelFilepath = QFileDialog.getOpenFileName(self, "Neural network model loading", "/home", "JSON Files (*.json)")[0]
        if weightsFilepath and modelFilepath :
            self.__network = neural_network_loader(weightsFilepath, modelFilepath)
            if self.__network != None :
                self.__networkLoaded = True
    
    # Load test file
    @Slot()
    def loadTestFile(self) :
        self.__testFilepath = QFileDialog.getOpenFileName(self, "Testing image loading", "/home", "Image Files (*.png *.jpg .bmp)")[0]
        if self.__testFilepath and self.__testFilepath[-3:] != "txt" :
            self.__image = QLabel(self)
            self.__imagetext = QLabel("Image overview", self)
            self.__imagetext.setFont(QFont("BebasNeue", 20, QFont.Bold))
            self.__imagetext.setAlignment(Qt.AlignCenter)
            self.__image.setAlignment(Qt.AlignCenter)
            self.__image.setObjectName("image")
            self.__image.setGeometry(800, 100, 350, 330)
            self.__imagetext.setGeometry(800, 400, 350, 100)
            self.__image.setPixmap(QPixmap(self.__testFilepath[:-4]).scaled(335, 315))
            self.__imagetext.show()
            self.__image.show()
    
    # Start testing
    @Slot()
    def startTest(self) :
        if self.__networkLoaded :
            if self.__testFilepath :
                if (self.__dico==None) :
                  classPath = QFileDialog.getOpenFileName(self, "Class dictionary", "/home", "Pickel Files (*.pickle)")[0]
                  if classPath :
                    pickle_in = open(classPath,"rb")
                    self.__dico= pickle.load(pickle_in)
                  else :
                    ret = QMessageBox.warning(self, "Class dictionary loading", "Cannot load the dictionary", QMessageBox.Ok)
                    return
                # Predicting an output
                key_list = list(self.__dico.keys())
                val_list = list(self.__dico.values())
                result = self.__network.validate(self.__testFilepath)
                res_str = ""
                res_str = str(key_list[val_list.index(result)])
                    
                self.__result = QLabel("Cette image represente : " + res_str, self)
                self.__result.setObjectName("result")
                self.__result.setFont(QFont("BebasNeue", 30, QFont.Bold))
                self.__result.setAlignment(Qt.AlignCenter)
                self.__result.setGeometry(700, 500, 500, 120)
                self.__result.show()
            else :
                ret = QMessageBox.warning(self, "Test file", "Cannot start the test, test file not loaded", QMessageBox.Ok)
                return
        else :
            ret = QMessageBox.warning(self, "Network not loaded", "Cannot start the test, network not loaded", QMessageBox.Ok)
            return
            

    # Initialiazing the window
    def __init__(self, *args, **kwargs):
        super(TestingWindow, self).__init__(*args, **kwargs)

        # Loading Fonts
        QFontDatabase.addApplicationFont("fonts/BebasNeue-Light.ttf")

        # Window Settings 
        self.setFixedSize(1280, 720)
        self.setWindowTitle("TestingWindow")
        background = QPixmap("images/testing")
        palette = QPalette()
        palette.setBrush(QPalette.Background, background)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Stylesheet Settings
        styleFile = QFile( "stylesheets/testing.qss" )
        styleFile.open( QFile.ReadOnly )
        style = str( styleFile.readAll() )
        self.setStyleSheet( style )

        # Title Settings
        self.__title = QLabel("Testing", self)
        self.__title.setFont(QFont("BebasNeue", 45, QFont.Bold))
        self.__title.setAlignment(Qt.AlignCenter)
        self.__title.setGeometry(0, 30, 1280, 100)

        # Button Settings 
        self.buttons = []
        for x in range(4) :
           self.buttons.append(x)
           self.buttons[x] = QPushButton(self)
           self.buttons[x].setCursor(Qt.PointingHandCursor)
           self.buttons[x].setObjectName("select")
           self.buttons[x].setFont(QFont("BebasNeue", 10, QFont.Bold))
           self.buttons[x].setGeometry(80, 100 + x * 120, 80, 80)

        # Connecting buttons
        self.buttons[0].clicked.connect(self.loadNetwork)
        self.buttons[1].clicked.connect(self.loadTestFile)
        self.buttons[2].clicked.connect(self.startTest)

        # Buttons Icons
        self.buttons[0].setIcon(QIcon("images/brain_icon"))
        self.buttons[0].setIconSize(QSize(35, 35))
        self.buttons[1].setIcon(QIcon("images/upload_icon"))
        self.buttons[1].setIconSize(QSize(35, 35))
        self.buttons[2].setIcon(QIcon("images/test_icon"))
        self.buttons[2].setIconSize(QSize(35, 35))
        self.buttons[3].setIcon(QIcon("images/goback_icon"))
        self.buttons[3].setIconSize(QSize(35, 35))
        
        # Return Button 
        self.buttons[3].setText("Back")
        self.buttons[3].setObjectName("retour")
        self.buttons[3].setGeometry(1100, 620, 120, 120)
        self.buttons[3].setIcon(QIcon("images/goback_icon"))
        self.buttons[3].setIconSize(QSize(35, 35))
        
        # Labels Settings
        self.__labels = []
        for x in range(3) :
            self.__labels.append(x)
            self.__labels[x] = QLabel(self)
            self.__labels[x].setGeometry(200, 110 + x * 120, 300, 80)
            self.__labels[x].setFont(QFont("BebasNeue", 20, QFont.Bold))
        self.__labels[0].setText("Load a neural network")
        self.__labels[1].setText("Load the test file")
        self.__labels[2].setText("Start the test")



# Main
#app = QApplication(sys.argv)
#window = TestingWindow()
#window.show()
#app.exec_()