import sys
from PySide2.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QPushButton
from PySide2.QtCore import Qt, QFile
from PySide2.QtGui import QPalette, QPixmap, QFont, QFontDatabase, QMovie

class MenuWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MenuWindow, self).__init__(*args, **kwargs)

        # Loading Fonts
        QFontDatabase.addApplicationFont("fonts/BebasNeue-Light.ttf")

        # Window Settings 
        self.setFixedSize(1280, 720)
        self.setWindowTitle("Systeme d'apprentissage base sur les reseaux de neurones")
        #background = QPixmap("images/menu")
        #palette = QPalette()
        #palette.setBrush(QPalette.Background, background)
        #self.setAttribute(Qt.WA_StyledBackground, True)
        #self.setPalette(palette)
        self.setAutoFillBackground(True)
        background = QLabel(self)
        movie = QMovie("images/menu")
        background.setMovie(movie)
        self.setCentralWidget(background)
        movie.start()

        # Stylesheet Settings
        styleFile = QFile( "stylesheets/menu.qss" )
        styleFile.open( QFile.ReadOnly )
        style = str( styleFile.readAll() )
        self.setStyleSheet( style )

        # Title Settings
        self.title = QLabel("Neural networks based learning system", self)
        self.title.setFont(QFont("BebasNeue", 45, QFont.Bold))
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
           self.buttons[x].setGeometry(400, 210 + x * 100, 550, 80)
           #self.buttons[x].clicked.connect(lambda : self.notifyMe(x))
        self.buttons[0].setText("Image preprocessing")        
        self.buttons[1].setText("Text preprocessing")        
        self.buttons[2].setText("Learning")
        self.buttons[3].setText("Testing")
        self.buttons[4].setText("Quit")

       

        



#app = QApplication(sys.argv)
#window = MenuWindow()
#window.show()
#app.exec_()