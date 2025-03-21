"""
I have two repositories.  Repo 1 contains 5 files relating to the cat trap game.  Three of them (CatGame.py, CatTrap.py, and hexutil.py)
are for the original version of the course.  Two of them (main.py and cat_trap_algorithms.py) are for the revised version of the course.
If you want to run the game, use my Repository 2 files (main.py and cat_trap_algorithms.py).  This code is consistent with the revised 
version of the course, which went live 3-11-2025.  With my 2 files (main.py and cat_trap_algorithms.py), I was able to find success. 
With my original 3 files (CatTrap.py, CatGame.py, hexutil.py), I was unable to run the game, so I spoke with Eduardo Corpeno via email.  
He advised that I may have success running the game by using this revised version, which included GitHub Codespaces.  

He mentioned that users were having trouble installing and setting up the environment, which is why he created the revised version.  
I ultimately found success getting the game to run by using GitHub Codespaces.  I was able to play the game.  In order to run the game,
one must run the main.py file and use Control + Shift + P, "Start Cat Trap Game".  You should run main.py from my Repository 2, 
which is entitled Week 1 Cat Game Code Assignment Repo 2.  Do this in Codespaces at Github.

CatGame.py in Repo 1 contains my evaluation function code.
"""

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *

from PyQt5.QtWidgets import *
from PyQt5.QtGui import  QFont

import math
from threading import Thread, Event

from CatGame import *
from hexutil import *
import random

TileRes = 35
stretch = 2.5 

class Level(object):

    def __init__(self, size):
        tiles = {}
        for tile in hexutil.origin.square_grid(size,size):
            tiles[tile] = '.'
        self.tiles = tiles
        self.seen_tiles = {}

    def get_tile(self, hexagon):
        return self.tiles.get(hexagon, '#')

    def set_tile(self, hexagon):
        return self.tiles.set(hexagon, '~')

    def get_seen_tile(self, hexagon):
        return self.seen_tiles.get(hexagon, ' ')

    def is_passable(self, hexagon):
        return self.get_tile(hexagon) not in '#~'

    def is_transparent(self, hexagon):
        return self.get_tile(hexagon) != '#'
 
    def update_fov(self, fov):
        for hexagon in fov:
            self.seen_tiles[hexagon] = self.get_tile(hexagon)


class GameWidget(QtWidgets.QWidget):
    

    _tile_brushes = {
            '.' : QtGui.QBrush(QtGui.QColor("yellow")),
            '~' : QtGui.QBrush(QtGui.QColor("blue")),
            '#' : QtGui.QBrush(QtGui.QColor("black")),
            }

    selected_hexagon = None

    def __init__(self, mainWidget, dim, *args, **kws):
        super().__init__(*args, **kws)

        self.setMouseTracking(True) 

        self.mainWidget = mainWidget
        self.dim = dim 
        self.level = Level(self.dim)
        self.center = ij_to_hex(self.dim//2,self.dim//2)
        self.hexgrid = hexutil.HexGrid(TileRes)
        self.restart()

        self.font = QtGui.QFont("Helvetica", 17)
        self.font.setStyleHint(QtGui.QFont.SansSerif)
        self.pen = QtGui.QPen()
        self.pen.setWidth(2)
        self.select_brush = QtGui.QBrush(QtGui.QColor(127, 127, 255, 127))
        self.unseen_brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 127))
        self.block_brush = QtGui.QBrush(QtGui.QColor("Gray"))
        self.cat_brush = QtGui.QBrush(QtGui.QColor("Orange"))
        self.update_fov()
        self.edit_mode=False

    def restart(self):
        self.cat = self.center
        self.game = Game(self.dim)
        self.blocks=[]
        
        self.game.init_random_blocks(self.cat)
        self.backannotate()
        self.repaint()

    def backannotate(self):
        self.cat = ij_to_hex(self.game.cat_i,self.game.cat_j)
        
        for i in range(self.game.size):
          for j in range(self.game.size):
              if self.game.tiles[i][j]==1:
                self.blocks.append(ij_to_hex(i,j))
    
    def setEditMode(self,state):
        self.edit_mode=state

    def update_fov(self):
        self.fov = self.center.field_of_view(transparent=self.level.is_transparent, max_distance=100)
        self.level.update_fov(self.fov)

    def hexagon_of_pos(self, pos):
        size = self.size()

        dx=TileRes+8
        dy=TileRes*round(math.sqrt(1.0/3.0))-2

        size = self.size()
        xc = size.width()//2
        yc = size.height()//2

        return self.center + self.hexgrid.hex_at_coordinate(pos.x() - xc, pos.y() - yc)

    def mousePressEvent(self, event):
        hexagon = self.hexagon_of_pos(event.pos())
        if self.edit_mode:

            hex_i, hex_j = hex_to_ij(hexagon)
            
            if hex_i>=self.dim or hex_j >=self.dim or \
               hex_i<0    or hex_j<0:
                return

            if self.cat==(-1,-1):                   
                if hexagon in self.blocks:          
                    self.blocks.remove(hexagon)
                self.game.tiles[hex_i][hex_j] = 6  
                self.game.cat_i = hex_i
                self.game.cat_j = hex_j
                self.cat = ij_to_hex(hex_i,hex_j)
                self.mainWidget.editCheckbox.setDisabled(False) 
            else:                                   
                if hexagon == self.cat:
                    self.game.tiles[hex_i][hex_j] = 0  
                    self.game.cat_i = -1
                    self.game.cat_j = -1
                    self.cat = ij_to_hex(-1,-1)
                    self.mainWidget.editCheckbox.setDisabled(True)

                elif hexagon in self.blocks:
                    self.blocks.remove(hexagon)
                    self.game.tiles[hex_i][hex_j]=0    
                else:
                    self.blocks.append(hexagon)
                    self.game.tiles[hex_i][hex_j]=1

            self.repaint()

        else:
            if hexagon == self.cat or hexagon in self.blocks:
                return

            hex_i, hex_j = hex_to_ij(hexagon)
            
            if hex_i>=self.dim or hex_j >=self.dim or \
               hex_i<0    or hex_j<0:
                return
            
            self.blocks.append(hexagon)
            
            self.game.tiles[hex_i][hex_j]=1
            self.game.print_tiles()
              

            cat_i, cat_j = self.game.cat_i,self.game.cat_j
            
            if cat_i==0 or cat_i==self.dim-1 or cat_j==0 or cat_j==self.dim-1:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Game Ended")
                msgBox.setText("The Cat Won... of course.")
                msgBox.exec()
                self.restart()
                self.repaint()
                return
        
            self.repaint()

            randcat = True if self.mainWidget.RCcheckbox.isChecked() else False
            ab = True if self.mainWidget.ABcheckbox.isChecked() else False
            DLS = True if self.mainWidget.DLScheckbox.isChecked() else False
            max_depth = int(self.mainWidget.depthText.text())
            ID = True if self.mainWidget.IDcheckbox.isChecked() else False
            alotted_time = float(self.mainWidget.timeText.text())

            newI,newJ = self.game.CustomCat(randcat,ab,DLS,max_depth,ID,alotted_time)
            
            print ("New cat coordinates:",newI,newJ)
            newHex=ij_to_hex(newI,newJ)
            if (newI == -1 and newJ == -1):
            	print("Time is up! Cat Removed.")

            if self.cat == newHex:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Game Ended")
                msgBox.setText("       You Won!!!       ")
                msgBox.exec()
                self.restart()
            else:
                self.game.tiles[cat_i][cat_j]=0
                self.cat=newHex
                self.game.tiles[newI][newJ]=6
                self.game.cat_i=newI
                self.game.cat_j=newJ
                
            self.repaint()


    def mouseMoveEvent(self, event):
        self.select_hexagon(event.pos())

    def select_hexagon(self, pos):
        hexagon = self.hexagon_of_pos(pos)
        
       if hexagon != self.selected_hexagon:
            self.selected_hexagon = hexagon
            self.repaint()
 
    def paintEvent(self, event):
        dx=TileRes+8
        dy=TileRes*round(math.sqrt(1.0/3.0))-2

        size = self.size()
        xc = size.width()//2
        yc = size.height()//2

        bbox = hexutil.Rectangle(-xc, -yc, 2*xc, 2*yc)

        hexgrid = self.hexgrid
        painter = QtGui.QPainter()
        painter.begin(self)
        try:
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor("darkGray"))

            painter.drawRect(xc-self.dim*dx, yc-self.dim*dy, self.dim*dx*2, self.dim*dy*2)
            painter.restore()

            painter.setPen(self.pen)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
            painter.setFont(QFont("Courier", 8, QFont.Bold))
            painter.translate(xc, yc)

            for hexagon in hexgrid.hexes_in_rectangle(bbox):
                polygon = QtGui.QPolygon([QtCore.QPoint(*corner) for corner in hexgrid.corners(hexagon)])
                hexagon2 = hexagon + self.center
                tile = self.level.get_seen_tile(hexagon2)
                if tile == ' ' or tile=='#':
                    continue
                painter.setBrush(self._tile_brushes[tile])
                painter.drawPolygon(polygon)

                if hexagon2 == self.selected_hexagon:
                    painter.setBrush(self.select_brush)
                    painter.drawPolygon(polygon)
                if hexagon2 in self.blocks:
                    painter.setBrush(self.block_brush)
                    painter.drawPolygon(polygon)
                    
                if hexagon2 == self.cat:
                    painter.setBrush(self.cat_brush)
                    rect = hexgrid.bounding_box(hexagon)
                    painter.drawPolygon(polygon)
                    rect = QtCore.QRectF(*rect) 
                    painter.drawText(rect, QtCore.Qt.AlignCenter, "/\\ /\\\n(=`I´=)")
                    
        finally:
            painter.end()

class MyTimer(Thread):
    def __init__(self, event, mainWidget, game):
        Thread.__init__(self)
        self.stopped = event

        self.start_time = time.time()
        self.deadline = self.start_time + mainWidget.cat_trap.alotted_time
        self.mainWidget=mainWidget

    def run(self,):
        while not self.stopped.wait(0.01):
            print("my thread")
            self.mainWidget.timeLeftLabel.setText(str('%.1f' % (self.deadline-time.time())))
            self.mainWidget.repaint()
            print(str('%.1f' % (self.deadline-time.time())))
        elapsed_time = (time.time() - self.start_time) * 1000
        print ("Elapsed time: %.3fms " % elapsed_time)


class MyWidget(QWidget):
    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent=parent)
        self.layoutUI()

    def layoutUI(self):

        self.mainLayout = QHBoxLayout(self)

        self.mainLayout.setContentsMargins(QMargins())  
        self.mainLayout.setSpacing(0)

        self.leftFrame = QFrame(self)
        self.leftFrame.setFrameShape(QFrame.StyledPanel)
        self.leftFrame.setFrameShadow(QFrame.Raised)
        
        
        self.rightFrame = QFrame(self)
        self.rightFrame.setFrameShape(QFrame.StyledPanel)
        self.rightFrame.setFrameShadow(QFrame.Raised)
        
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.spaceLayout = QVBoxLayout()
        
        height = 30


        self.dimLabel = QLabel("The Cat Trap Game\nis an NxN HexGrid\n\nEnter an odd number for N:")
        self.dimLabel.setFixedSize(200,2.5*height)

        self.dimText = QLineEdit("7")
        self.dimText.setFixedSize(100,height)
        self.dimText.textEdited.connect(self.updateDimText)

        self.timeLabel = QLabel("Deadline in seconds:")
        self.timeLabel.setFixedSize(200,height)

        self.timeText = QLineEdit("5")
        self.timeText.setFixedSize(100,height)

        self.RCcheckbox = QCheckBox("Random Cat")
        self.RCcheckbox.stateChanged.connect(self.updateRCcheckbox)
        self.RCcheckbox.setFixedWidth(200)
        
        self.ABcheckbox = QCheckBox("Alpha-Beta Pruning")
        self.ABcheckbox.setChecked(True)
        self.ABcheckbox.setFixedWidth(200)
        

        self.DLScheckbox = QCheckBox("Limited Depth:", self)
        self.DLScheckbox.stateChanged.connect(self.updateDLScheckbox)
        self.DLScheckbox.setFixedWidth(200)

        self.depthText = QLineEdit("4")
        self.depthText.setDisabled(True)
        self.depthText.setFixedSize(100,height)
        

        self.IDcheckbox = QCheckBox("Iterative Deepening")
        self.IDcheckbox.stateChanged.connect(self.updateIDcheckbox)
        self.IDcheckbox.setChecked(True)
        self.RCcheckbox.setFixedWidth(200)
        
        
        self.spaceLabel1 = QLabel(" ")
        self.spaceLabel1.setFixedSize(100,height/3)
        self.spaceLabel2 = QLabel(" ")
        self.spaceLabel2.setFixedSize(10,height/3)

        self.start_button = QPushButton()
        self.start_button.setFixedSize(200, 2*height)
        self.start_button.setText("Start New Game")
        self.start_button.setStyleSheet("background-color: white;")
        self.start_button.clicked.connect(self.on_click)
   
        self.hresizer = QLabel(" ")
        self.hresizer.setFixedWidth(100)

        self.editCheckbox = QCheckBox("Edit Mode")
        self.editCheckbox.setFixedSize(200,height)
        self.editCheckbox.stateChanged.connect(self.updateEditCheckbox)
   
        self.cat_trap = GameWidget(self,7)

        self.cat_trap.setFixedWidth(TileRes*self.cat_trap.dim*stretch)
        self.cat_trap.setFixedHeight(TileRes*self.cat_trap.dim*2)	

       
        self.spaceLayout.addWidget(self.spaceLabel2)

        self.leftLayout.addWidget(self.spaceLabel1)
        self.leftLayout.addWidget(self.cat_trap)
                
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.start_button)
        self.rightLayout.addWidget(self.dimLabel)
        self.rightLayout.addWidget(self.dimText)
        self.rightLayout.addWidget(self.timeLabel)
        self.rightLayout.addWidget(self.timeText)
        self.rightLayout.addWidget(self.spaceLabel1)        
        self.rightLayout.addWidget(QHLine())
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.RCcheckbox)
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(QHLine())
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.ABcheckbox)
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(QHLine())
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.DLScheckbox)
        self.rightLayout.addWidget(self.depthText)
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(QHLine())
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.IDcheckbox)   
        self.rightLayout.addWidget(self.spaceLabel1)  
        self.rightLayout.addWidget(QHLine())
        self.rightLayout.addWidget(self.spaceLabel1)
        self.rightLayout.addWidget(self.editCheckbox)
        self.rightLayout.addWidget(self.hresizer)

        self.mainLayout.addLayout(self.spaceLayout)
        self.mainLayout.addLayout(self.leftLayout)
        self.mainLayout.addLayout(self.rightLayout)
        
        self.mainLayout.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(self.mainLayout)

    def updateDimText(self, text):
        try: 
            if int(text)%2==0:
                self.dimText.setText(str(int(text)+1))
        finally:
            return

    def updateRCcheckbox(self, state):
        if state == Qt.Checked:
            self.DLScheckbox.setChecked(False)
            self.DLScheckbox.setDisabled(True)
            self.IDcheckbox.setChecked(False)
            self.IDcheckbox.setDisabled(True)
            self.ABcheckbox.setChecked(False)
            self.ABcheckbox.setDisabled(True)
        else:
            self.DLScheckbox.setChecked(True)
            self.DLScheckbox.setDisabled(False)
            self.IDcheckbox.setDisabled(False)
            self.ABcheckbox.setDisabled(False)

    def updateDLScheckbox(self, state):
        if state == Qt.Checked:
            self.depthText.setDisabled(False)
            self.IDcheckbox.setChecked(False)
        else:
            self.depthText.setDisabled(True)
            
    def updateIDcheckbox(self, state):
        if state == Qt.Checked:
            self.DLScheckbox.setChecked(False)

    def updateEditCheckbox(self, state):
        if state == Qt.Checked:
            self.cat_trap.setEditMode(True)
        else:
            self.cat_trap.setEditMode(False)

    def on_click(self):
        print('New Game Started')
        print("Hexgrid dimensions:",self.dimText.text(),"x",self.dimText.text())

        self.leftLayout.removeWidget(self.cat_trap)
        self.cat_trap.deleteLater()
        self.cat_trap = None

        self.cat_trap = GameWidget(self,int(self.dimText.text()))
        self.cat_trap.setFixedWidth(TileRes*self.cat_trap.dim*stretch)
        self.cat_trap.setFixedHeight(TileRes*self.cat_trap.dim*2)	
        self.leftLayout.addWidget(self.cat_trap)
        self.editCheckbox.setDisabled(False)
        self.editCheckbox.setChecked(False)

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def main():
    app = QApplication(sys.argv)
    w = MyWidget()
    w.setWindowTitle("Trap the Cat")
    w.resize(TileRes*w.cat_trap.dim*stretch+200,TileRes*w.cat_trap.dim*2)
    w.show()
    sys.exit(app.exec_())

main()
