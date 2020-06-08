import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QTableWidgetItem, QTableWidget, QMainWindow, QPushButton, QFileDialog, QLineEdit, QCheckBox, QVBoxLayout, QFormLayout, QGroupBox, QComboBox, QMessageBox, QRadioButton, QHBoxLayout
from PySide2.QtCore import Qt, QFile, QObject, QSize, Signal, Slot, QTimer
from PySide2.QtGui import QPalette, QPixmap, QFont, QFontDatabase, QIcon, QDoubleValidator, QIntValidator, QPainter
from PySide2.QtCharts import QtCharts
from LayerButton import LayerButton
from network import Network
from fullyConnected_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from activations import *
from losses import *
from convolutional_layer import ConvLayer
from flatten_layer import FlattenLayer
from keras.datasets import mnist
from keras.utils import np_utils
from preprocessing_text import fill_missing_values, min_max_normalize_dataset
from Iostream import save_matrix_neural_network, manual_save_model_neural_network

class TrainingWindow(QWidget):

    # Dataset
    __dataset = None

    # Netwok model
    __modelList = []

    # Start training
    @Slot()
    def startTraining(self) :
        
        # Get split value
        split = 1 - float((self.__datasetSplitComboBox.currentIndex() + 1)/10.0)

        # Get split method
        if self.__datasetSplitRandom.isChecked() :
            ((x_train, y_train), (x_test, y_test)) = self.network.random_split(self.__dataset, split)
        elif self.__datasetSplitRegular.isChecked() :
            ((x_train, y_train), (x_test, y_test)) = self.network.regular_split(self.__dataset, split)


        # Get epochs number
        if not self.__epochsLineEdit.text() :
            ret = QMessageBox.warning(self, "Epochs number", "Please enter the number of epochs", QMessageBox.Ok)
            return
        else :
            epochs = int(self.__epochsLineEdit.text())
            self.__xAxis.setRange(0, epochs)

        # Get learning rate value
        if not self.__learningRateLineEdit.text() :
            ret = QMessageBox.warning(self, "Learning rate", "Please select a learning rate", QMessageBox.Ok)
            return 
        else :
            learning_rate = float(self.__learningRateLineEdit.text().replace(",", "."))
            if not learning_rate : 
                ret = QMessageBox.warning(self, "Learning rate", "The learning rate cannot be equal to zero", QMessageBox.Ok)
                return

        # Get learning rate mode
        if self.__learningRateCheckBox.isChecked() :
            mode = 2
        else :
            mode = 1

        # Save before training
        ret = QMessageBox.question(self, "Network save", "Would you like to save the network before the training starts?", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes :
            save_matrix_neural_network(self.network, self.networkName)
            manual_save_model_neural_network(self.network, self.networkName)
            QMessageBox.information(self, "Network save", "Network successfully saved !", QMessageBox.Ok)
        
        # Clearing the graph
        self.__series.clear()

        # Starting training
        length = len(x_train)
        for i in range(epochs):
            err = 0
            training_accuracy = 0
            l_rate = self.network.Learning_rate_schedule(mode,epochs,epochs-i+1,learning_rate)
            for j in range(length):
                outputs = x_train[j]
                for layer in self.network.layers:
                    outputs = layer.forward_propagation(outputs)
                err += self.network.loss(y_train[j], outputs)
                training_accuracy = training_accuracy + self.network.verification_of_prediction(x_train,y_train,j)
                error = self.network.loss_prime(y_train[j], outputs)
                for layer in reversed(self.network.layers):
                    error = layer.backward_propagation(error, l_rate)
            err = err / length
            training_accuracy = training_accuracy / float(length)
            self.__epochNumberLabel.setText("Epoch : " + str(i+1) + "/" + str(epochs))
            self.__trainingAccuracyLabel.setText("Taux de precision : " + str(training_accuracy*100) + "%")
            # Appending values to the chart
            self.__series.append(i, training_accuracy*100)
            self.__chartView.repaint()
            # Auto saving network
            save_matrix_neural_network(self.network, self.networkName + "_auto")
            manual_save_model_neural_network(self.network, self.networkName + "_auto")

        # Saving trained network
        ret = QMessageBox.question(self, "Network save", "Would you like to save the trained network? ", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes :
            save_matrix_neural_network(self.network, self.networkName)
            manual_save_model_neural_network(self.network, self.networkName)
            QMessageBox.information(self, "Network save", "Network successfully saved !", QMessageBox.Ok)
            
        # Evaluate network and show confusion matrix
        (self.test_accuracy, self.matrix) = self.network.evaluate(x_test, y_test)
        self.__confusionMatrixButton.show()

    
    # Showing the confusion matrix
    @Slot()
    def showStats(self) :
        # Creating matrix window 
        self.matrixWindow = QMainWindow()
        self.matrixWindow.setFixedSize(640, 480)

        key_list = list(self.__classes.keys())
        val_list = list(self.__classes.values())

        # Creating matrix table 
        self.matrixTable = QTableWidget(len(self.matrix) + 1, len(self.matrix[0]) + 1)
        for i in range(len(self.matrix)) :
            self.matrixTable.setItem(i+1, 0, QTableWidgetItem(str(key_list[val_list.index(i)])))
            self.matrixTable.setItem(0, i+1, QTableWidgetItem(str(key_list[val_list.index(i)])))

        for i in range(len(self.matrix)) :
            for j in range(len(self.matrix[0])) :
                self.matrixTable.setItem(i+1, j+1, QTableWidgetItem(str(self.matrix[i][j])))

        # Printing test accuracy
        self.matrixLabel = QLabel("Test accuracy : " + str(self.test_accuracy * 100) + "%", self)
        self.matrixLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))

        # Matrix window layout 
        self.matrixLayout = QVBoxLayout()
        self.matrixLayout.addWidget(self.matrixTable)
        self.matrixLayout.addWidget(self.matrixLabel)

        # Matrix window groupbox
        self.matrixGroupBox = QGroupBox(self.matrixWindow)
        self.matrixGroupBox.setLayout(self.matrixLayout)

        # Showing the matrix window
        self.matrixWindow.setCentralWidget(self.matrixGroupBox)
        self.matrixWindow.show()

    def __init__(self,  ds, classes, model, created, *args, **kwargs):
        super(TrainingWindow, self).__init__(*args, **kwargs)

        if created :
            # Initialising network
            self.network = Network()
            self.network.use(mean_squared_error, mean_squared_error_prime)
        else :
            self.network = model[0]

        
        # Getting inputs and outputs
        self.__dataset = ds
        self.__classes = classes
        #fill_missing_values(self.__dataset)
        #min_max_normalize_dataset(self.__dataset)
        ((x_train, y_train), (x_test, y_test)) = self.network.regular_split(self.__dataset, 0.5)
        
        # Getting inputs
        if len(x_train.shape) == 2 :
            inputs = x_train.shape[1]
        else :
            inputs = x_train.shape[1:]
            first = inputs[0]
            second = inputs[1]
            third = inputs[2]
        
        # Getting expected outputs
        expected_output = y_train.shape[1]

        # Getting network name
        self.networkName = model[1]

        
        
        if created :
           # Getting model list
            self.__modelList = model[0]
            self.__modelList[0].setOutput(inputs)

            for i in range(1, len(self.__modelList)) :
                    # Getting the layer name
                    name = self.__modelList[i].text()[:len(self.__modelList[i].text())- 6]
                    activation = None
                    activ_prime = None
                    # Getting the activation function
                    if self.__modelList[i].activation() == 0 :
                        activation = sigmoid
                        activ_prime = sigmoid_prime
                    elif self.__modelList[i].activation() == 1 :
                        activation = tanh
                        activ_prime = tanh_prime
                    elif self.__modelList[i].activation() == 2 :
                        activation = rectified_linear_unit
                        activ_prime = rectified_linear_unit_prime
                    elif self.__modelList[i].activation() == 3 :
                        activation = softmax
                        activ_prime = softmax_prime
                    # Adding layer to the network 
                    if  name == "Dense" : 
                        if self.__modelList[i-1].text()[:2] == "Fl" :
                            self.network.add(FullyConnectedLayer(first * second * third, self.__modelList[i].output()))
                            self.network.add(ActivationLayer(activation, activ_prime))
                        else :
                            self.network.add(FullyConnectedLayer(self.__modelList[i - 1].output(), self.__modelList[i].output()))
                            self.network.add(ActivationLayer(activation, activ_prime))
                    elif name == "Flatten" :
                        self.network.add(FlattenLayer())
                    elif name == "Convolutional" :
                        self.network.add(ConvLayer((first, second, third), (self.__modelList[i].kernelRows, self.__modelList[i].kernelColumns), 1))
                        self.network.add(ActivationLayer(activation, activ_prime))
                        first = first - self.__modelList[i].kernelRows + 1
                        second = second - self.__modelList[i].kernelColumns + 1

            self.network.add(FullyConnectedLayer(self.__modelList[len(self.__modelList) - 1].output(), expected_output))
            self.network.add(ActivationLayer(sigmoid, sigmoid_prime))



        # Loading Fonts
        QFontDatabase.addApplicationFont("fonts/BebasNeue-Light.ttf")

        # Window Settings 
        self.setFixedSize(1280, 720)
        self.setWindowTitle("Training window")
        #background = QPixmap("images/menu")
        #palette = QPalette()
        #palette.setBrush(QPalette.Background, background)
        #self.setAttribute(Qt.WA_StyledBackground, True)
        #self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Stylesheet Settings
        styleFile = QFile( "stylesheets/training.qss" )
        styleFile.open( QFile.ReadOnly )
        style = str( styleFile.readAll() )
        self.setStyleSheet( style )

        # Title Settings
        self.title = QLabel("Training", self)
        self.title.setFont(QFont("BebasNeue", 30, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setGeometry(600, 10, 300, 120)

        # Epochs line edit settings 
        self.__epochsLineEdit = QLineEdit(self)
        self.__epochsLineEdit.setValidator(QIntValidator(0, 100000, self))

        # Epochs label settings
        self.__epochsLabel = QLabel("Epoch number", self)
        self.__epochsLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))

        # Learning rate line edit settings
        self.__learningRateLineEdit = QLineEdit(self)
        self.__learningRateLineEdit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))

        # Learning rate label settings
        self.__learningRateLabel = QLabel("Learning rate", self)
        self.__learningRateLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))

        # Learning rate checkboxsettings (auto or not)
        self.__learningRateCheckBox = QCheckBox("Auto adjustment", self)
        self.__learningRateCheckBox.setFont(QFont("BebasNeue", 15, QFont.Bold))

        # Dataset split settings label
        self.__datasetSplitLabel = QLabel("Dataset split percentage", self)
        self.__datasetSplitLabel.setFont((QFont("BebasNeue", 20, QFont.Bold)))

        # Dataset split mode buttons 
        self.__datasetSplitRegular = QRadioButton("Regular split")
        self.__datasetSplitRandom = QRadioButton("Random split")

        # Dataset split mode buttons groupbox
        self.__datasetSplitModeButtonsLayout = QHBoxLayout(self)
        self.__datasetSplitModeButtonsGroupBox = QGroupBox(self)
        self.__datasetSplitModeButtonsGroupBox.setObjectName("setting")
        self.__datasetSplitModeButtonsLayout.addWidget(self.__datasetSplitRegular)
        self.__datasetSplitModeButtonsLayout.addWidget(self.__datasetSplitRandom)
        self.__datasetSplitModeButtonsGroupBox.setLayout(self.__datasetSplitModeButtonsLayout)
        self.__datasetSplitRegular.setChecked(True)

        # Dataset split combo box settings
        self.__datasetSplitComboBox = QComboBox(self)
        self.__datasetSplitComboBox.addItems(['90% - 10%', '80% - 20%' , '70% - 30%', '60% - 40%'])

        # Dataset split form layout settings
        self.__datasetSplitLayout = QFormLayout(self)
        self.__datasetSplitGroupBox = QGroupBox(self)
        self.__datasetSplitGroupBox.setObjectName("setting")
        self.__datasetSplitLayout.addWidget(self.__datasetSplitLabel)
        self.__datasetSplitLayout.addWidget(self.__datasetSplitComboBox)
        self.__datasetSplitGroupBox.setLayout(self.__datasetSplitLayout)

        # Epochs form layout settings
        self.__epochsFormLayout = QFormLayout(self)
        self.__epochsGroupBox = QGroupBox(self)
        self.__epochsGroupBox.setObjectName("setting")
        self.__epochsFormLayout.addWidget(self.__epochsLabel)
        self.__epochsFormLayout.addWidget(self.__epochsLineEdit)
        self.__epochsGroupBox.setLayout(self.__epochsFormLayout)

        # Learning rate form layout settings 
        self.__learningRateFormLayout = QFormLayout(self)
        self.__learningRateGroupBox = QGroupBox(self)
        self.__learningRateGroupBox.setObjectName("setting")
        self.__learningRateFormLayout.addWidget(self.__learningRateLabel)
        self.__learningRateFormLayout.addWidget(self.__learningRateCheckBox)
        self.__learningRateFormLayout.addWidget(self.__learningRateLineEdit)
        self.__learningRateGroupBox.setLayout(self.__learningRateFormLayout)

        # Epochs number label 
        self.__epochNumberLabel = QLabel("Epoch : ", self)
        self.__epochNumberLabel.setFont((QFont("BebasNeue", 15, QFont.Bold)))

        # Training accuracy label
        self.__trainingAccuracyLabel = QLabel("Accuracy : ", self)
        self.__trainingAccuracyLabel.setFont((QFont("BebasNeue", 15, QFont.Bold)))

        # Training stats layout 
        self.__trainingStatsLayout = QVBoxLayout(self)
        self.__trainingStatsGroupBox = QGroupBox(self)
        self.__trainingStatsLayout.addWidget(self.__epochNumberLabel)
        self.__trainingStatsLayout.addWidget(self.__trainingAccuracyLabel)
        self.__trainingStatsGroupBox.setLayout(self.__trainingStatsLayout)
        self.__trainingStatsGroupBox.setGeometry(1000, -30, 300, 150)

        # Training button settings 
        self.__trainingButton = QPushButton("Start", self)
        self.__trainingButton.setCursor(Qt.PointingHandCursor)
        self.__trainingButton.setFont((QFont("BebasNeue", 30, QFont.Bold)))
        self.__trainingButton.clicked.connect(self.startTraining)

        # Go back button
        self.goBackButton = QPushButton("Back", self)
        self.goBackButton.setObjectName("retour")

        # Customising go back button
        self.goBackButton.setCursor(Qt.PointingHandCursor)
        self.goBackButton.setIcon(QIcon("images/goback_icon"))
        self.goBackButton.setIconSize(QSize(30, 30))
        self.goBackButton.setFont(QFont("BebasNeue", 20, QFont.Bold))

        # Confusion matrix button
        self.__confusionMatrixButton = QPushButton("Show confusion matrix", self)
        self.__confusionMatrixButton.setCursor(Qt.PointingHandCursor)
        self.__confusionMatrixButton.setFont((QFont("BebasNeue", 17, QFont.Bold)))
        self.__confusionMatrixButton.clicked.connect(self.showStats)
        self.__confusionMatrixButton.setGeometry(420, 20, 250, 80)
        self.__confusionMatrixButton.hide()

        # Parameters group box settings
        self.__parametersGroupBox = QGroupBox("Training parameters", self)
        self.__parametersGroupBox.setObjectName("parameters")
        self.__parametersLayout = QVBoxLayout(self)
        self.__parametersLayout.addWidget(self.__epochsGroupBox)
        self.__parametersLayout.addWidget(self.__datasetSplitGroupBox)
        self.__parametersLayout.addWidget(self.__datasetSplitModeButtonsGroupBox)
        self.__parametersLayout.addWidget(self.__learningRateGroupBox)
        self.__parametersLayout.addWidget(self.__trainingButton)
        self.__parametersLayout.addWidget(self.goBackButton)
        self.__parametersGroupBox.setLayout(self.__parametersLayout)
        self.__parametersGroupBox.setGeometry(0, 0, 400, 720)

        # Chart axis settings
        self.__xAxis = QtCharts.QValueAxis()
        self.__xAxis.setRange(0, 5)

        self.__yAxis = QtCharts.QValueAxis()
        self.__yAxis.setRange(0, 100)

        # Chart settings
        self.__series = QtCharts.QLineSeries()
        self.__chart = QtCharts.QChart()
        self.__chart.addAxis(self.__xAxis, Qt.AlignBottom)
        self.__chart.addAxis(self.__yAxis, Qt.AlignLeft)
        self.__chart.addSeries(self.__series)
        self.__series.attachAxis(self.__xAxis)
        self.__series.attachAxis(self.__yAxis)
        self.__chart.setTitle("Accuracy")
        self.__chartView = QtCharts.QChartView(self.__chart)
        self.__chartView.setRenderHint(QPainter.Antialiasing)

        # Chart layout settings
        self.__chartLayout = QVBoxLayout(self)
        self.__chartGroupBox = QGroupBox(self)
        self.__chartGroupBox.setObjectName("chart")
        self.__chartLayout.addWidget(self.__chartView)
        self.__chartGroupBox.setLayout(self.__chartLayout)
        self.__chartGroupBox.setGeometry(390, 100, 900, 600)


        # Update timer settings
        #self.__timer = QTimer(self)
        #self.__timer.timeout.connect(self.autoSave)
        #self.__timer.start(1000)

    
#app = QApplication(sys.argv)
#window = TrainingWindow()
#window.show()
#app.exec_()