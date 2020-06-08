from PySide2.QtWidgets import QPushButton, QWidget
from PySide2.QtCore import Qt, QMimeData, QObject, Signal, Slot, QRect
from PySide2.QtGui import QDrag, QDropEvent
import sys

class ConnectButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super(ConnectButton,self).__init__(*args, **kwargs)
        self.__connected = False

    def setConnected(self, b):
        self.__connected = b
    
    def isConnected(self) :
        return self.__connected

class LayerButton(QPushButton):
  
    def __init__(self, *args, **kwargs):
        super(LayerButton, self).__init__(*args, **kwargs)

        # Input connecting button
        self.inputButton = ConnectButton(self)
        self.inputButton.setObjectName("input")
        self.inputButton.move(-17, 20)

        # Output connecting button
        self.outputButton = ConnectButton(self)
        self.outputButton.setObjectName("output")
        self.outputButton.move(165, 20)

        # Layer button attributes
        self.__output = 0
        self.__activation = 0
        self.kernelRows = 0
        self.kernelColumns = 0


    def mouseMoveEvent(self, e):

        if e.buttons() != Qt.LeftButton:
            return

        mimeData = QMimeData()

        self.__cursor_x = e.pos().x()
        self.__cursor_y = e.pos().y()

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        self.hide()
        drag.setPixmap(self.grab(QRect(2, 2, self.width() - 3.5, self.height() - 3.5)))
        drag.setHotSpot(e.pos())
        dropAction = drag.exec_(Qt.MoveAction)
        self.show()
    
    # Getters
    def getCursorX(self) :
        return self.__cursor_x
    def getCursorY(self) :
        return self.__cursor_y
    def output(self) :
        return self.__output
    def activation(self) :
        return self.__activation
    
    def setOutput(self, o) :
        self.__output = o
    def setActivation(self, a) :
        self.__activation = a
