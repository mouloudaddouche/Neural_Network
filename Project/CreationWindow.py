import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QFormLayout, QComboBox, QGraphicsScene, QGraphicsView, QMessageBox
from PySide2.QtCore import Qt, QFile, QSize, QObject, Signal, Slot, QPoint
from PySide2.QtGui import QPalette, QPixmap, QFont, QFontDatabase, QIcon, QDropEvent, QIntValidator, QPaintEvent, QPainter, QPen, QColor, QCursor
from LayerButton import LayerButton
import pandas as pd

class CreationWindow(QWidget) :

    # Layer buttons vector
    __layerVector = []

    # Deleting mode
    __deleteMode = False

    # Connecting mode
    __connectingMode = False

    # Layer parameters box
    __parametersShown = False

    # Last layer that opened the parameters window
    __currentLayer = None

    # Last layer that opened connecting mode
    __connectingLayer = None

    # Lines
    __lines = []

    # Done signal
    done = Signal()

    # Dataset
    dataset = None

    @Slot()
    def closeParameters(self):
        # Closing parameters window
        if self.__parametersShown :
            self.__parametersShown = False
            self.__layerParametersGroupBox.hide()
            self.__layerOutputLineEdit.setText(str(""))
            self.__currentLayer = None

    @Slot()
    def acceptParameters(self) :
        # Setting new layer parameters
        if self.__layerOutputLineEdit.text() :
            self.__currentLayer.setOutput(int(self.__layerOutputLineEdit.text()))
        self.__currentLayer.setActivation(self.__activationMenu.currentIndex())
        if self.__kernelRowsLineEdit.text() :
            self.__currentLayer.kernelRows = int(self.__kernelRowsLineEdit.text())
        if self.__kernelColumnLineEdit.text() :
            self.__currentLayer.kernelColumns = int(self.__kernelColumnLineEdit.text())

    @Slot() 
    def layerAction(self):
        # Deleting the layer if delete mode is on
        if self.__deleteMode :
            address = id(self.sender())
            self.sender().deleteLater()
            line1 = -1
            line2 = -1
            # Removing layer from current layers list
            for i in range(len(self.__layerVector)) :
                if id(self.__layerVector[i]) == address :
                    del self.__layerVector[i]
                    break
            # Removing layer from lines list 
            for i in range(len(self.__lines)) :
                if id(self.__lines[i][0].parentWidget()) == address :
                    self.__lines[i][1].setConnected(False)
                    line1 = i
                elif id(self.__lines[i][1].parentWidget())== address :
                    self.__lines[i][0].setConnected(False)
                    line2 = i
            if line1 != -1 :
                del self.__lines[line1]
            if line2 != -1 :
                del self.__lines[line2]
        # Showing parameters window if delete mode is off
        elif self.sender().text()[:2] != "In" and self.sender().text()[:2] != "Ou" and self.sender().text()[:2] != "Fl" :
            self.__currentLayer = self.sender()
            if self.sender().text()[:13] == "Convolutional" :
                self.__kernelRowsGroupBox.show()
                self.__kernelColumnGroupBox.show()
                self.__activationGroupBox.show()
                self.__layerOutputGroupBox.hide()
            else :
                self.__layerOutputGroupBox.show()
                self.__activationGroupBox.show()
                self.__kernelRowsGroupBox.hide()
                self.__kernelColumnGroupBox.hide()
            self.__layerOutputLineEdit.setText(str(self.sender().output()))
            self.__activationMenu.setCurrentIndex(self.sender().activation())
            if not self.__parametersShown :
                self.__layerParametersGroupBox.show()
                self.__layerParametersGroupBox.raise_()
                self.__parametersShown = True


    @Slot()
    def reset(self) :
        # Deleting all layers in the list and clearing it
        if self.__layerVector :
            for i in range(len(self.__layerVector)) :
                self.__layerVector[i].deleteLater()
            del self.__layerVector[:]
        # Clearing lines list
        del self.__lines[:]
        # Hiding the parameters window 
        self.__layerParametersGroupBox.hide()
            
    @Slot()
    def confirm(self) :
        # Checking and confirming model
        self.modelList = []
        input = False
        tmp = 0
        # Checking network name 
        if not self.networkName.text() :
            ret = QMessageBox.warning(self, "Network name", "Enter a name for the neural network", QMessageBox.Ok)
            return
        # If no layers are connected
        if not self.__lines :
            ret = QMessageBox.warning(self, "Invalid model", "No layer connected", QMessageBox.Ok)
            return
        # If one or more layers are not connected
        elif len(self.__lines) != len(self.__layerVector) - 1 :
            ret = QMessageBox.warning(self, "Invalid model", "Some layers are unconnected", QMessageBox.Ok)
            return
        else :
            # Finding the input layer
            for line in self.__lines :
                if line[0].parentWidget().text()[:5] == "Input" :
                    self.modelList.append(line[0].parentWidget())
                    self.modelList.append(line[1].parentWidget())
                    tmp = id(line[1].parentWidget())
                    input = True
            # If there is no input 
            if not input :
                ret = QMessageBox.warning(self, "Invalid model", "No input layer found", QMessageBox.Ok)
                return
            # Adding all the layers to a list
            while len(self.modelList) != len(self.__lines) + 1 :
                for line in self.__lines :
                    if id(line[0].parentWidget()) == tmp :
                        self.modelList.append(line[1].parentWidget())
                        tmp = id(line[1].parentWidget())
            
            # If there is no output
            if self.modelList[-1].text()[:6] != "Output" :
                ret = QMessageBox.warning(self, "Invalid model", "No output layer found", QMessageBox.Ok)
                return
            del self.modelList[-1]
            # Checking layers
            c = 0
            for x in self.modelList :
                if isinstance(self.dataset, pd.DataFrame) :
                    if x.text()[:2] == "Co" :
                        ret = QMessageBox.warning(self, "Convolutional layer", "Cannot put a convolutional layer. Dataset is alphanumerical", QMessageBox.Ok)
                        return
                    elif x.text()[:2] == "Fl" :
                        ret = QMessageBox.warning(self, "Flatten layer", "Cannot put a flatten layer. Dataset is alphanumerical", QMessageBox.Ok)
                        return
                if x.text()[:2] == "Co" and self.modelList[c-1].text()[:2] == "De" :
                    ret = QMessageBox.warning(self, "Dense layer", "A dense layer cannot be followed by a convolutional layer", QMessageBox.Ok)
                    return
                if x.text()[:2] == "Fl" and self.modelList[c-1].text()[:2] != "Co" :
                    ret = QMessageBox.warning(self, "Flatten layer", "A flatten layer must always be preceded by a convolutional layer", QMessageBox.Ok)
                    return
                if x.text()[:2] == "Co" and (self.modelList[c+1].text()[:2] != "Fl" and self.modelList[c+1].text()[:2] != "Co") : 
                    ret = QMessageBox.warning(self, "Convolutional layer", "A convolutional layer must always be followed by a flatten or convolutional layer", QMessageBox.Ok)
                    return
                if x.text()[:2] == "De" and not x.output() :
                    ret = QMessageBox.warning(self, "Layer output number", "Please select an output number for each layer", QMessageBox.Ok)
                    return
                if x.text()[:2] == "Co" and (not x.kernelColumns or not x.kernelRows) :
                    ret = QMessageBox.warning(self, "Kernel size", "Please select a kernel size for convolutional layers", QMessageBox.Ok)
                    return
                c = c + 1

            # Emit signal to switch the window
            self.done.emit()


    @Slot()
    def connectLayer(self) :
        # Activating connecting mode (line following cursor)
        if not self.__connectingMode :
            if not self.sender().isConnected() :
                self.__connectingMode = True
                self.__connectingLayer = self.sender()
        else :
            parent_1 = id(self.__connectingLayer.parentWidget())
            parent_2 = id(self.sender().parentWidget())
            connectingLayerType = self.__connectingLayer.objectName()
            senderLayerType = self.sender().objectName()
            # Checking if the connection works
            # Can't connect an already connected layer
            # parent_1 != parent_2 : Can't connect the input and output of the same layer
            # Can only connect an Input with an Output or an Output with an Input
            if not self.sender().isConnected() and (parent_1 != parent_2) and ((connectingLayerType == "input" and senderLayerType == "output") or (connectingLayerType == "output" and senderLayerType == "input")) :
                # Connecting the two buttons
                if connectingLayerType == "input" :
                    self.__lines.append([self.sender(), self.__connectingLayer])
                else :
                    self.__lines.append([self.__connectingLayer, self.sender()])
                self.__connectingLayer.setConnected(True)
                self.sender().setConnected(True)
                # Deactivating connecting mode and reseting last layer clicked
                self.__connectingMode = False
                self.__connectingLayer = None

            

    @Slot()
    def createLayer(self):
        # Creating the layer 
        layer = LayerButton(self.sender().text(), self)
        layer.setFont(QFont("BebasNeue", 20, QFont.Bold))
        layer.setGeometry(400, 220, 180, 80)
        layer.setObjectName("layer")     
        # Layer icon settings
        layer.setIcon(QIcon(self.sender().icon()))
        layer.setIconSize(QSize(30, 30))
        # Hiding the input/output button on an Input/Output Layer
        if layer.text()[:5] == "Input" :
            layer.inputButton.hide()
        elif layer.text()[:6] == "Output" :
            layer.outputButton.hide()
        # Setting the cursor accordingly
        if not self.__deleteMode :
            layer.setCursor(Qt.PointingHandCursor)  
        else :  
            layer.setCursor(Qt.CrossCursor) 
        # Connecting the layer to its slots
        layer.clicked.connect(self.layerAction)
        layer.inputButton.clicked.connect(self.connectLayer)    
        layer.outputButton.clicked.connect(self.connectLayer)        
        # Showing the layer
        layer.show()
        # Adding the layer to the list of current layers
        self.__layerVector.append(layer) 
        # Raising the parameters groupbox on top of the screen
        self.__layerParametersGroupBox.raise_()

    # Initialiazing the window
    def __init__(self, dataset, *args, **kwargs):
        super(CreationWindow, self).__init__(*args, **kwargs)

        # Initializing dataset
        self.dataset = dataset

        # Loading Fonts
        QFontDatabase.addApplicationFont("fonts/BebasNeue-Light.ttf")

        # Accepting drag & drops
        self.setAcceptDrops(True)

        # Window Settings 
        self.setFixedSize(1280, 720)
        self.setWindowTitle("Neural network creation")
        background = QPixmap("images/grid")
        palette = QPalette()
        palette.setBrush(QPalette.Background, background)
        self.setPalette(palette)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        # Enabling mouse tracking
        self.setMouseTracking(True)

        # Creating graphics scene
        #self.__scene = QGraphicsScene(300, 120, 980, 600, self)
        #self.__scene.addLine(400, 500, 600, 650, QPen(Qt.yellow, 10))
        #self.__liveFeedView = QGraphicsView()
        #self.__liveFeedView.setScene(self.__scene)

        # Stylesheet Settings
        styleFile = QFile( "stylesheets/creation.qss" )
        styleFile.open( QFile.ReadOnly )
        style = str( styleFile.readAll() )
        self.setStyleSheet( style )

        # Network name line edit
        self.networkName = QLineEdit(self)
        self.networkName.setPlaceholderText("Enter neural network name")

        # Netwok name label 
        self.__networkNameLabel = QLabel("Neural network name : ", self)
        self.__networkNameLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))
        self.__networkNameLabel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.__networkNameLabel.setBuddy(self.networkName)

        # Accept/Reset buttons
        self.__topRightButtons = []
        for x in range(2) :
           self.__topRightButtons.append(x)
           self.__topRightButtons[x] = QPushButton("Bouton ici", self)
           self.__topRightButtons[x].setCursor(Qt.PointingHandCursor)
           self.__topRightButtons[x].setIconSize(QSize(35, 35))
           self.__topRightButtons[x].setFont(QFont("BebasNeue", 10, QFont.Bold))


        # Customising accept/reset buttons
        self.__topRightButtons[0].setText("Reset")
        self.__topRightButtons[0].setIcon(QIcon("images/reset_icon"))
        self.__topRightButtons[1].setText("Confirm")
        self.__topRightButtons[1].setIcon(QIcon("images/check_icon"))

        # Connecting accept/reset buttons
        self.__topRightButtons[0].clicked.connect(self.reset)
        self.__topRightButtons[1].clicked.connect(self.confirm)

        # Go back button
        self.goBackButton = QPushButton("Back", self)
        self.goBackButton.setObjectName("retour")

        # Connecting go back button
        self.goBackButton.clicked.connect(self.reset)

        # Customising go back button
        self.goBackButton.setCursor(Qt.PointingHandCursor)
        self.goBackButton.setIcon(QIcon("images/goback_icon"))
        self.goBackButton.setIconSize(QSize(30, 30))
        self.goBackButton.setFont(QFont("BebasNeue", 20, QFont.Bold))

        # Layer selection buttons
        self.__layerButtons = []
        for x in range(5) :
           self.__layerButtons.append(x)
           self.__layerButtons[x] = QPushButton(self)
           self.__layerButtons[x].setCursor(Qt.PointingHandCursor)
           self.__layerButtons[x].setFont(QFont("BebasNeue", 10, QFont.Bold))
           self.__layerButtons[x].clicked.connect(self.createLayer)

        # Layer buttons names
        self.__layerButtons[0].setText("Input layer")        
        self.__layerButtons[1].setText("Output layer")        
        self.__layerButtons[2].setText("Dense layer")        
        self.__layerButtons[3].setText("Flatten layer")
        self.__layerButtons[4].setText("Convolutional layer")

        # Layer buttons icons
        for x in range(5) :
            icon = "images/layer_icon_"
            self.__layerButtons[x].setIcon(QIcon(icon + str(x)))
            self.__layerButtons[x].setIconSize(QSize(45, 45))

        # Top buttons layout settings
        self.__buttonLayout = QHBoxLayout(self)
        self.__buttonGroupBox = QGroupBox(self)
        self.__buttonGroupBox.setGeometry(780, -15, 500, 120)
        self.__buttonLayout.addWidget(self.__topRightButtons[0])
        self.__buttonLayout.addWidget(self.__topRightButtons[1])
        self.__buttonGroupBox.setLayout(self.__buttonLayout)

        # Network name form layout settings
        self.__networkNameLayout = QFormLayout(self)
        self.__networkNameGroupBox = QGroupBox(self)
        self.__networkNameGroupBox.setGeometry(300, -15, 480, 120)
        self.__networkNameLayout.addWidget(self.__networkNameLabel)
        self.__networkNameLayout.addWidget(self.networkName)
        self.__networkNameGroupBox.setLayout(self.__networkNameLayout)

        # Layer buttons layout settings
        self.__layerButtonLayout = QVBoxLayout(self)
        self.__layerButtonGroupBox = QGroupBox("Layer selection", self)
        self.__layerButtonGroupBox.setGeometry(0, -15, 300, 735)
        for x in range(5) :
            self.__layerButtonLayout.addWidget(self.__layerButtons[x])
        self.__layerButtonLayout.addWidget(self.goBackButton)
        self.__layerButtonGroupBox.setLayout(self.__layerButtonLayout)

        # Parameters window settings
        # Layer output label
        self.__layerOutputLabel = QLabel("Output number", self)
        self.__layerOutputLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))
        # Layer output line edit
        self.__layerOutputLineEdit = QLineEdit(self)
        self.__layerOutputLineEdit.setValidator(QIntValidator(0, 1000, self))
        # Layer output form settings 
        self.__layerOutputLayout = QFormLayout(self)
        self.__layerOutputGroupBox = QGroupBox(self)
        self.__layerOutputLayout.addWidget(self.__layerOutputLabel)
        self.__layerOutputLayout.addWidget(self.__layerOutputLineEdit)
        self.__layerOutputGroupBox.setLayout(self.__layerOutputLayout)
        # Activation function label
        self.__activationLabel = QLabel("Activation function", self)
        self.__activationLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))
        # Activation function menu 
        self.__activationMenu = QComboBox(self)
        self.__activationMenu.addItems(['Sigmoid', 'Tanh' , 'Rectified Linear Unit', 'Softmax'])
        # Activation function form settings
        self.__activationLayout = QFormLayout(self)
        self.__activationGroupBox = QGroupBox(self)
        self.__activationLayout.addWidget(self.__activationLabel)
        self.__activationLayout.addWidget(self.__activationMenu)
        self.__activationGroupBox.setLayout(self.__activationLayout)
        # Close window button 
        self.__closeButton = QPushButton(self)
        self.__closeButton.setObjectName("close")
        self.__closeButton.setCursor(Qt.PointingHandCursor)
        self.__closeButton.setIcon(QIcon("images/close_icon"))
        self.__closeButton.setIconSize(QSize(35, 35))
        self.__closeButton.clicked.connect(self.closeParameters)
        # Accept changes button
        self.__acceptButton = QPushButton(self)
        self.__acceptButton.setObjectName("accept")
        self.__acceptButton.setCursor(Qt.PointingHandCursor)
        self.__acceptButton.setIcon(QIcon("images/accept_icon"))
        self.__acceptButton.setIconSize(QSize(35, 35))
        self.__acceptButton.clicked.connect(self.acceptParameters)
        # Close/Accept buttons layout
        self.__bottomButtonsLayout = QHBoxLayout(self)
        self.__bottomButtonsGroupBox = QGroupBox(self)
        self.__bottomButtonsLayout.addWidget(self.__closeButton)
        self.__bottomButtonsLayout.addWidget(self.__acceptButton)
        self.__bottomButtonsGroupBox.setLayout(self.__bottomButtonsLayout)
        # Kernel rows label
        self.__kernelRowsLabel = QLabel("Kernel rows", self)
        self.__kernelRowsLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))
        # Kernel rows line edit 
        self.__kernelRowsLineEdit = QLineEdit(self)
        self.__kernelRowsLineEdit.setValidator(QIntValidator(0, 1000, self))
        # Kernel rows form layout 
        self.__kernelRowsLayout = QFormLayout(self)
        self.__kernelRowsLayout.addWidget(self.__kernelRowsLabel)
        self.__kernelRowsLayout.addWidget(self.__kernelRowsLineEdit)
        # Kernel rows group box
        self.__kernelRowsGroupBox = QGroupBox(self)
        self.__kernelRowsGroupBox.setLayout(self.__kernelRowsLayout)
        self.__kernelRowsGroupBox.hide()
        # Kernel columns label
        self.__kernelColumnLabel = QLabel("Kernel columns", self)
        self.__kernelColumnLabel.setFont(QFont("BebasNeue", 20, QFont.Bold))
        # Kernel columns line edit 
        self.__kernelColumnLineEdit = QLineEdit(self)
        self.__kernelColumnLineEdit.setValidator(QIntValidator(0, 1000, self))
        # Kernel columns form layout 
        self.__kernelColumnLayout = QFormLayout(self)
        self.__kernelColumnLayout.addWidget(self.__kernelColumnLabel)
        self.__kernelColumnLayout.addWidget(self.__kernelColumnLineEdit)
        # Kernel columns group box
        self.__kernelColumnGroupBox = QGroupBox(self)
        self.__kernelColumnGroupBox.setLayout(self.__kernelColumnLayout)
        self.__kernelColumnGroupBox.hide()
        # Layer parameters group box
        self.__layerParametersGroupBox = QGroupBox(self)    
        self.__layerParametersGroupBox.setObjectName("parameters")  
        self.__layerParametersGroupBox.setGeometry(960, 88, 320, 550)
        self.__layerParametersLayout = QVBoxLayout(self)   
        self.__layerParametersLayout.addWidget(self.__layerOutputGroupBox)
        self.__layerParametersLayout.addWidget(self.__activationGroupBox)
        self.__layerParametersLayout.addWidget(self.__kernelRowsGroupBox)
        self.__layerParametersLayout.addWidget(self.__kernelColumnGroupBox)
        self.__layerParametersLayout.addWidget(self.__bottomButtonsGroupBox)
        self.__layerParametersGroupBox.setLayout(self.__layerParametersLayout)
        self.__layerParametersGroupBox.hide()
        self.__layerParametersGroupBox.raise_()

    # Overloading dragEnterEvent method
    def dragEnterEvent(self, e):  
        e.accept()        

    # Overloading dropEvent method
    def dropEvent(self, e):
        # Getting the event source
        button = QDropEvent.source(e)
        # Calculating the new coordinates
        new_x = e.pos().x() - button.getCursorX()
        new_y = e.pos().y() - button.getCursorY()
        # Moving the button if it is still in frame
        if new_x > 350 and new_y > 120 and new_x < 1100 and new_y < 640:
            position = QPoint(new_x, new_y)
            button.move(position)
            e.setDropAction(Qt.MoveAction)
            e.accept()
    
    # Overloading keyPressEvent to activate/deactivate deleting mode
    def keyPressEvent(self, ev) :
        # Activating delete mode when pressing delete key
        if not self.__deleteMode and ev.key() == Qt.Key_Delete :
            self.setCursor(Qt.CrossCursor)
            self.__deleteMode = True
            for i in range(len(self.__layerVector)) :
                    self.__layerVector[i].setCursor(Qt.CrossCursor)
        # Deactivating delete mode when pressing escape key
        elif self.__deleteMode and ev.key() == Qt.Key_Escape :
            self.setCursor(Qt.ArrowCursor)
            self.__deleteMode = False
            for i in range(len(self.__layerVector)) :
                    self.__layerVector[i].setCursor(Qt.PointingHandCursor)
        # Deactivating connecting mode when pressing escape key
        if self.__connectingMode and ev.key() == Qt.Key_Escape :
            self.__connectingMode = False


    # Overloading mouse press event to unfocus QLineEdit
    def mousePressEvent(self, event):
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            focused_widget.clearFocus()
        QWidget.mousePressEvent(self, event)
        if self.__deleteMode :
            i = 0
            for line in self.__lines :
                # Getting the line equation
                slope = (line[0].y() + line[0].parentWidget().y() - line[1].y() - line[1].parentWidget().y()) / float(line[0].x() + line[0].parentWidget().x() - line[1].x() - line[1].parentWidget().x())
                point_slope = line[0].y() + line[0].parentWidget().y() + 20 - slope * (line[0].x() + line[0].parentWidget().x() + 20)
                # Checking if the coordinates of the mouse click are on the line
                if event.pos().y() < int(slope * event.pos().x() + point_slope + 3) and  event.pos().y() > int(slope * event.pos().x() + point_slope - 3) :
                    # Delete the line here
                    self.__lines[i][0].setConnected(False)
                    self.__lines[i][1].setConnected(False)
                    self.__lines.remove(line)
                i = i + 1
    
    # Overloading mouse move event to change the cursor if its out of frame
    def mouseMoveEvent(self, ev) :
        if self.__deleteMode :
            if ev.pos().x() < 325 or ev.pos().y() < 120 :
                self.setCursor(Qt.ArrowCursor)
            else :
                self.setCursor(Qt.CrossCursor)
        if self.__connectingMode :
            if ev.pos().x() < 340 or ev.pos().y() < 130 :
                self.__connectingMode = False
    
    # Drawing lines
    def paintEvent(self, e) :
        painter = QPainter(self)
        painter.setPen(QPen(QColor(145, 18, 9), 10))
        if self.__connectingMode :
            layerPoint = QPoint(self.__connectingLayer.x() + self.__connectingLayer.parentWidget().x() + 20, self.__connectingLayer.y() + self.__connectingLayer.parentWidget().y() + 20)
            painter.drawLine(layerPoint, self.mapFromGlobal(QCursor.pos()))
            self.update()
        if self.__lines :
            for line in self.__lines :
                point1 = QPoint(line[0].x() + line[0].parentWidget().x() + 20, line[0].y() + line[0].parentWidget().y() + 20)
                point2 = QPoint(line[1].x() + line[1].parentWidget().x() + 20, line[1].y() + line[1].parentWidget().y() + 20)
                self.update()
                painter.drawLine(point1, point2)


# Main
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window = CreationWindow()
    window.show()
    app.exec_()


