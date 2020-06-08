import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox
from PySide2.QtCore import Qt, QFile, QObject, Signal, Slot
from PySide2.QtGui import QPalette, QPixmap, QFont, QFontDatabase
from Iostream import neural_network_loader, dataset_csv_loader, dataset_arff_loader, load_dataset


class ChoiceWindow(QWidget):

    # Load dataset 
    @Slot()
    def loadDataset(self) :
        datasetPath = QFileDialog.getOpenFileName(self, "Dataset loading", "/home", "Pickle files (*.pickle)")[0]
        classPath = QFileDialog.getOpenFileName(self, "Class file loading", "/home", "Pickle files (*.pickle)")[0]
        if datasetPath and classPath:
            if not load_dataset(datasetPath, classPath) : 
                QMessageBox.warning(self, "Dataset loading", "An error has occured while loading the dataset", QMessageBox.Ok)
            (self.dataset, self.classes) = load_dataset(datasetPath, classPath) 
            
            self.datasetLoaded = True
            self.datasetLabel.show()


    # Load network
    @Slot()
    def loadNetwork(self) :
        weightsFilepath = QFileDialog.getOpenFileName(self, "Neural network weights loading", "/home", "HDF5 Files (*.h5)")[0]
        modelFilepath = QFileDialog.getOpenFileName(self, "Neural network model loading", "/home", "JSON Files (*.json)")[0]
        self.networkName = weightsFilepath[weightsFilepath.rfind("/") + 1 : weightsFilepath.find(".")]
        if "_auto" in self.networkName :
            self.networkName = self.networkName.replace("_auto", "")
        if weightsFilepath and modelFilepath :
            self.network = neural_network_loader(weightsFilepath, modelFilepath)
            if not self.network :
                QMessageBox.warning(self, "Network loading", "An error has occured while loading the neural network", QMessageBox.Ok)
            else : 
                self.networkLoaded = True
                self.networkLabel.show()



    def __init__(self, *args, **kwargs):
        super(ChoiceWindow, self).__init__(*args, **kwargs)

        # Boolean indicating if the network is correctly loaded
        self.networkLoaded = False

        # Boolean indicating if the dataset is correctly loaded
        self.datasetLoaded = False

        # Loading Fonts
        QFontDatabase.addApplicationFont("fonts/BebasNeue-Light.ttf")

        # Window Settings 
        self.setFixedSize(1280, 720)
        self.setWindowTitle("ChoiceWindow")
        background = QPixmap("images/choice")
        palette = QPalette()
        palette.setBrush(QPalette.Background, background)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Stylesheet Settings
        styleFile = QFile( "stylesheets/choice.qss" )
        styleFile.open( QFile.ReadOnly )
        style = str( styleFile.readAll() )
        self.setStyleSheet( style )

        # Title Settings
        self.title = QLabel("Choose an option", self)
        self.title.setFont(QFont("BebasNeue", 60, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setGeometry(0, 60, 1280, 120)

        # Button Settings 
        self.buttons = []
        for x in range(5) :
           self.buttons.append(x)
           self.buttons[x] = QPushButton(self)
           self.buttons[x].setCursor(Qt.PointingHandCursor)
           self.buttons[x].setObjectName("select")
           self.buttons[x].setFont(QFont("BebasNeue", 15, QFont.Bold))
           self.buttons[x].setGeometry(430, 220 + x * 120, 450, 100)
        self.buttons[0].setText("Load a dataset")    
        self.buttons[0].clicked.connect(self.loadDataset)    

        self.buttons[1].setText("Load a neural network")
        self.buttons[1].clicked.connect(self.loadNetwork)

        self.buttons[2].setText("Create a neural network")
        self.buttons[3].setText("Start training")

        self.buttons[4].setText("Back")
        self.buttons[4].setGeometry(1100, 620, 100, 80)

        # Dataset label
        self.datasetLabel = QLabel(self)
        self.icone = QPixmap("images/check_icon")
        self.datasetLabel.setPixmap(self.icone.scaled(60, 60))
        self.datasetLabel.setAlignment(Qt.AlignCenter)
        self.datasetLabel.setGeometry(900, 240, 60, 60)
        self.datasetLabel.hide()

        # Network label
        self.networkLabel = QLabel(self)
        self.icone2 = QPixmap("images/check_icon")
        self.networkLabel.setPixmap(self.icone2.scaled(60, 60))
        self.networkLabel.setAlignment(Qt.AlignCenter)
        self.networkLabel.setGeometry(900, 350, 60, 60)
        self.networkLabel.hide()

       

#    
#app = QApplication(sys.argv)
#window = ChoiceWindow()
#window.show()
#app.exec_()