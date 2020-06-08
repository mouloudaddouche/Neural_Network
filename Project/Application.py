import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QMessageBox
from PySide2.QtCore import Qt, QFile, QSize, QObject, Signal, Slot
from PySide2.QtGui import QFontDatabase
from MenuWindow import MenuWindow
from TestingWindow import TestingWindow
from CreationWindow import CreationWindow
from ChoiceWindow import ChoiceWindow
from TrainingWindow import TrainingWindow
from PreprocessingImageWindow import PreprocessingImageWindow
from PreprocessingTextWindow import PreprocessingTextWindow


class Application(QMainWindow):

    # Booleans indicating which window is opened
    __isTesting = False
    __isTraining = False
    __isCreation = False
    __isPreProcessingText = False
    __isPreProcessingImage = False
    __isChoice = False

    @Slot()
    def goToTraining(self) :
        if self.__isCreation :
            self.__isCreation = False
            self.__isTraining = True
            self.__trainingWindow = TrainingWindow(self.__creationWindow.dataset, self.__choiceWindow.classes, (self.__creationWindow.modelList, self.__creationWindow.networkName.text()), True)
            self.__trainingWindow.goBackButton.clicked.connect(self.backtoCreation)
            self.__windows.addWidget(self.__trainingWindow)
            self.__windows.setCurrentIndex(3)

        elif self.__isChoice :
            if not self.__choiceWindow.datasetLoaded :
                ret = QMessageBox.warning(self, "Dataset", "Veuillez charger un dataset", QMessageBox.Ok)
                return
            if not self.__choiceWindow.networkLoaded :
                ret = QMessageBox.warning(self, "Reseau de neurones", "Veuillez charger un reseau de neurones ou en creer un nouveau", QMessageBox.Ok)
                return
            self.__isTraining = True
            self.__isChoice = False
            self.__trainingWindow = TrainingWindow(self.__choiceWindow.dataset, self.__choiceWindow.classes, (self.__choiceWindow.network, self.__choiceWindow.networkName), False)
            self.__trainingWindow.goBackButton.clicked.connect(self.backToChoice)
            self.__windows.addWidget(self.__trainingWindow)
            self.__windows.setCurrentIndex(2)

    @Slot()
    def goToTesting(self) :
        self.__isTesting = True
        self.__testingWindow = TestingWindow()
        self.__windows.addWidget(self.__testingWindow)
        self.__windows.setCurrentIndex(1)
        self.__testingWindow.buttons[3].clicked.connect(self.backToMenu)

    @Slot()
    def goToCreation(self) :
        if not self.__choiceWindow.datasetLoaded :
            ret = QMessageBox.warning(self, "Dataset", "Veuillez d'abord charger un dataset", QMessageBox.Ok)
            return
        self.__isChoice = False
        if self.__choiceWindow.networkLoaded :
            ret = QMessageBox.warning(self, "Reseau de neurones", "Un reseau de neurones est deja charge", QMessageBox.Ok)
            return
        self.__isCreation = True
        self.__creationWindow = CreationWindow(self.__choiceWindow.dataset)
        self.__windows.addWidget(self.__creationWindow)
        self.__windows.setCurrentIndex(2)
        self.__creationWindow.goBackButton.clicked.connect(self.backToChoice)
        self.__creationWindow.done.connect(self.goToTraining)

    @Slot()
    def goToChoice(self) :
        self.__isChoice = True
        self.__choiceWindow = ChoiceWindow()
        self.__windows.addWidget(self.__choiceWindow)
        self.__windows.setCurrentIndex(1)
        self.__choiceWindow.buttons[2].clicked.connect(self.goToCreation)
        self.__choiceWindow.buttons[3].clicked.connect(self.goToTraining)
        self.__choiceWindow.buttons[4].clicked.connect(self.backToMenu)

    @Slot()
    def backtoCreation(self) :
        self.__isCreation = True
        self.__isTraining= False
        self.__windows.setCurrentIndex(2)
        self.__windows.removeWidget(self.__trainingWindow)
        self.__trainingWindow.deleteLater()

    @Slot()
    def goToPreprocessingText(self) :
        self.__isPreProcessingText = True
        self.__preProcessingTextWindow = PreprocessingTextWindow()
        self.__preProcessingTextWindow.pushButton_return.clicked.connect(self.backToMenu)
        self.__windows.addWidget(self.__preProcessingTextWindow)
        self.__windows.setCurrentIndex(1)
    
    @Slot()
    def goToPreprocessingImage(self) :
        self.__isPreProcessingImage = True
        self.__preProcessingImageWindow = PreprocessingImageWindow()
        self.__preProcessingImageWindow.pushButton_return.clicked.connect(self.backToMenu)
        self.__windows.addWidget(self.__preProcessingImageWindow)
        self.__windows.setCurrentIndex(1)

    @Slot()
    def backToMenu(self) :
        self.__windows.setCurrentIndex(0)
        if self.__isTesting :
            self.__isTesting = False
            self.__windows.removeWidget(self.__testingWindow)
            self.__testingWindow.deleteLater()
        elif self.__isPreProcessingText :
            self.__isPreProcessingText = False
            self.__windows.removeWidget(self.__preProcessingTextWindow)
            self.__preProcessingTextWindow.deleteLater()
        elif self.__isPreProcessingImage :
            self.__isPreProcessingImage = False
            self.__windows.removeWidget(self.__preProcessingImageWindow)
            self.__preProcessingImageWindow.deleteLater()
        elif self.__isChoice :
            self.__isChoice = False
            self.__windows.removeWidget(self.__choiceWindow)
            self.__choiceWindow.deleteLater()
            self.__choiceWindow = None

    @Slot()
    def backToChoice(self) :
        self.__isChoice = True
        self.__windows.setCurrentIndex(1)
        if self.__isCreation :
            self.__isCreation = False
            self.__windows.removeWidget(self.__creationWindow)
            self.__creationWindow.deleteLater()
        elif self.__isTraining :
            self.__isTraining = False
            self.__windows.removeWidget(self.__trainingWindow)
            self.__trainingWindow.deleteLater()
        

    def __init__(self, *args, **kwargs) :
        super(Application, self).__init__(*args, **kwargs)

        # Initializing menu window
        self.__menuWindow = MenuWindow()
        self.__windows = QStackedWidget(self)
        self.__windows.addWidget(self.__menuWindow)
        self.__windows.setCurrentIndex(0)
        self.setCentralWidget(self.__windows)

        # Connecting buttons
        self.__menuWindow.buttons[0].clicked.connect(self.goToPreprocessingImage)
        self.__menuWindow.buttons[1].clicked.connect(self.goToPreprocessingText)
        self.__menuWindow.buttons[2].clicked.connect(self.goToChoice)
        self.__menuWindow.buttons[3].clicked.connect(self.goToTesting)
        self.__menuWindow.buttons[4].clicked.connect(QApplication.quit)
        

# Main
app = QApplication(sys.argv)
window = Application()
window.show()
app.exec_()