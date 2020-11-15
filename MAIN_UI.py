import PyQt5
from PyQt5 import QtWidgets,QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QTextEdit, QGroupBox, QMessageBox, QVBoxLayout, QSizePolicy 
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import pandas as pd
import numpy as np
import matplotlib,csv
matplotlib.use('Qt5Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure



#-------------SETTING THE GRAPH FOR OLD ALGORITHM-----------------
class MplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


#-------------SETTING THE GRAPH FOR NEW ALGORITHM-----------------
class newMplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(newMplCanvas, self).__init__(fig)







# It is considered a good tone to name classes in CamelCase.
class MyFirstGUI(QtWidgets.QDialog): 
    
    def __init__(self):
        # Initializing QDialog and locking the size at a certain value
        super(MyFirstGUI, self).__init__()
        self.setFixedSize(1200, 670)
        self.fileName = "test.csv"
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("HYBRID LOGISTIC REGRESSION ON TWITTER POST")


        #--------------DIVIDER--------------------
        self.divider_top = QGroupBox(self)
        self.divider_top.resize(1200,4)
        self.divider_top.move(1,75)


        self.divider_sideleft = QGroupBox(self)
        self.divider_sideleft.resize(4,500)
        self.divider_sideleft.move(500,75)


        self.divider_sideright = QGroupBox(self)
        self.divider_sideright.resize(4,500)
        self.divider_sideright.move(700,75)

        #---------------BUTTONS-----------------------------

        self.sentiment = QPushButton('SENTIMENT', self)
        self.sentiment.move(700,20)
        self.sentiment.resize(100,50)
        self.sentiment.clicked.connect(self.senti)





        self.generate = QPushButton('GENERATE', self)
        self.generate.setToolTip('Generate The Algorithm')
        self.generate.move(820,20)
        self.generate.resize(100,50)
        #self.generate.setEnabled(False)
        self.generate.clicked.connect(self.on_click)

        self.generatetwo = QPushButton('GENERATE', self)
        self.generatetwo.setToolTip('Generate The Algorithm')
        self.generatetwo.move(820,20)
        self.generatetwo.resize(100,50)
        self.generatetwo.setEnabled(False)
        self.generatetwo.setHidden(True)
        self.generatetwo.clicked.connect(self.on_show)

        self.clear = QPushButton('CLEAR', self)
        self.clear.move(940,20)
        self.clear.resize(100,50)
        self.clear.setEnabled(False)
        self.clear.clicked.connect(self.on_clear)

        self.exitui = QPushButton('EXIT', self)
        self.exitui.move(1050,20)
        self.exitui.resize(100,50)
        self.exitui.clicked.connect(self.terminate)

        


        


        
        #----GRAPH BOX FOR OLD ALGORITHM--------
        self.lbl_numb = QLabel('OLD ALGORITHM', self)
        self.lbl_numb.move(150,100)
        self.oldalgo_GraphBox = QGroupBox(self)
        self.oldalgo_GraphBox.resize(350,250)
        self.oldalgo_GraphBox.move(30,120)

        self.lbl_accurateold = QLabel(self)
        self.lbl_accurateold.setText('ACCURACY: ')
        self.lbl_accurateold.move(170,380)


        self.lbl_confuseold = QLabel(self)
        self.lbl_confuseold.setText('CONFUSION MATRIX')
        self.lbl_confuseold.move(390,200)


        


        #-------------------------------------------------


        #----GRAPH BOX FOR NEW ALGORITHM--------
        self.newalgo_GraphBox = QtWidgets.QGroupBox('', self)
        self.newalgo_GraphBox.resize(350,250)
        self.newalgo_GraphBox.move(830,120)

        self.lbl_numb = QLabel(self)
        self.lbl_numb.setText('NEW ALGO')
        self.lbl_numb.move(950,100)

        self.lbl_accuratenew = QLabel(self)
        self.lbl_accuratenew.setText('ACCURACY: ')
        self.lbl_accuratenew.move(950,380)

        self.lbl_confusenew = QLabel(self)
        self.lbl_confusenew.setText('CONFUSION MATRIX')
        self.lbl_confusenew.move(720,200)


        

        #--------------------RESULT FOR OLD ALGO------------------------
        

        self.lbl_tally_positive = QLabel(self)
        self.lbl_tally_positive.setText('POSITIVE: ')
        self.lbl_tally_positive.move(50,430)

        self.lbl_tally_neutral = QLabel(self)
        self.lbl_tally_neutral.setText('NEUTRAL: ')
        self.lbl_tally_neutral.move(180,430)

        self.lbl_tally_negative = QLabel(self)
        self.lbl_tally_negative.setText('NEGATIVE: ')
        self.lbl_tally_negative.move(280,430)

        self.lbl_tally_overall = QLabel(self)
        self.lbl_tally_overall.setText('OVERALL RESULT: ')
        self.lbl_tally_overall.move(150,470)

        #--------------------RESULT FOR NEW ALGO------------------------
        

        self.lbl_tally_hybpositive = QLabel(self)
        self.lbl_tally_hybpositive.setText('POSITIVE: ')
        self.lbl_tally_hybpositive.move(800,430)

        self.lbl_tally_hybneutral = QLabel(self)
        self.lbl_tally_hybneutral.setText('NEUTRAL: ')
        self.lbl_tally_hybneutral.move(950,430)

        self.lbl_tally_hybnegative = QLabel(self)
        self.lbl_tally_hybnegative.setText('NEGATIVE: ')
        self.lbl_tally_hybnegative.move(1080,430)

        self.lbl_tally_hyboverall = QLabel(self)
        self.lbl_tally_hyboverall.setText('OVERALL RESULT: ')
        self.lbl_tally_hyboverall.move(940,470)

        #--------------------------------------------




        #----------------------TABLE FOR THE LIST OF GATHERED TWEETS-------------------
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.model)
        self.tableView.resize(1100, 150)
        self.tableView.move(40,500)
        #-------------------------------------------------------------------





















    #---------------GENERATING THE ALGORITHM-----------------------------------
    def senti(self,event):
        import PREPREOCESS_SENTIMENT
        self.sentiment_result = QMessageBox()
        self.sentiment_result.setWindowTitle("LOGISTIC REGRESSION")
        self.sentiment_result.setText("SENTIMENT TWEETS COMPLETE")
        self.sentiment_result.setIcon(QMessageBox.Information)
        self.y = self.sentiment_result.exec_()

    def on_click(self, event):
        import BASELINE as twit #CODE FOR BASELINE 
        import re_log_Hyb as hyb #CODE FOR HYBRID
        
        #-----------------BASELINE RESULTS------------------------
        
        #self.vbox = QVBoxLayout()
        #self.canvas = twit.logicreg(self, width=8, height=6)
        #self.canvas.move(250, 100)
        final_accurate = str(twit.percentage)
        final_pos = str(twit.posi)
        final_neu = str(twit.neut)
        final_neg = str(twit.nega)
        final_cm  = str(twit.confuse)
        final_report = str(twit.reports)
        post_accurate = 'ACCURACY: ' +  final_accurate
        post_pos = 'POSITIVE: ' + final_pos
        post_neu = 'NEUTRAL: ' + final_neu
        post_neg = 'NEGATIVE: ' + final_neg
        post_overall = 'OVERALL RESULT: ' + twit.overall
        post_cm = 'CONFUSION MATRIX' + '\n' + final_cm
        #print(post_cm)

       


        #-----------------HYBRID RESULTS------------------------

        final_hyb_accurate = str(hyb.percentage)
        final_hyb_cm  = str(hyb.confuse)
        final_hyb_report = str(hyb.reports)
        post_hyb_accurate = 'ACCURACY: ' +  final_hyb_accurate
        post_hyb_cm = 'CONFUSION MATRIX' + '\n' + final_hyb_cm 

        final_hybpos = str(hyb.posi)
        final_hybneu = str(hyb.neut)
        final_hybneg = str(hyb.nega)
        post_hybpos = 'POSITIVE: ' + final_hybpos
        post_hybneu = 'NEUTRAL: ' + final_hybneu
        post_hybneg = 'NEGATIVE: ' + final_hybneg
        post_hyboverall = 'OVERALL RESULT: ' + hyb.overall



        #-----PLOT GRAPH FOR OLD-----------------
        ploted = twit.plots
        self.m = MplCanvas(self, width=5, height=4)
        self.m.axes.plot(ploted)
        self.layout_oldalgo = QtWidgets.QVBoxLayout()
        self.layout_oldalgo.addWidget(self.m)
        self.oldalgo_GraphBox.setLayout(self.layout_oldalgo)
        #----------------GRAPH FOR NEW-----------------------------
        hyb_ploted = hyb.plots
        self.n = newMplCanvas(self, width=5, height=4)
        self.n.axes.plot(hyb_ploted)
        self.layout_newalgo = QtWidgets.QVBoxLayout()
        self.layout_newalgo.addWidget(self.n)
        self.newalgo_GraphBox.setLayout(self.layout_newalgo)

        



        #------------------TEXT CHANGED WITH THE RESULT----------------------------
       
        self.lbl_accuratenew.setText(post_hyb_accurate)
        self.lbl_accuratenew.adjustSize()

        self.lbl_accurateold.setText(post_accurate)
        self.lbl_accurateold.adjustSize()


        self.lbl_tally_positive.setText(post_pos)
        self.lbl_tally_positive.adjustSize()

        
        self.lbl_tally_neutral.setText(post_neu)
        self.lbl_tally_neutral.adjustSize()

        
        self.lbl_tally_negative.setText(post_neg)
        self.lbl_tally_negative.adjustSize()

        self.lbl_tally_overall.setText(post_overall)
        self.lbl_tally_overall.adjustSize()


        #-------------HYBRID---------------------
        self.lbl_tally_hybpositive.setText(post_hybpos)
        self.lbl_tally_hybpositive.adjustSize()

        
        self.lbl_tally_hybneutral.setText(post_hybneu)
        self.lbl_tally_hybneutral.adjustSize()

        
        self.lbl_tally_hybnegative.setText(post_hybneg)
        self.lbl_tally_hybnegative.adjustSize()

        self.lbl_tally_hyboverall.setText(post_hyboverall)
        self.lbl_tally_hyboverall.adjustSize()

        self.lbl_confuseold.setText(post_cm)
        self.lbl_confuseold.adjustSize()

        self.lbl_confusenew.setText(post_hyb_cm)
        self.lbl_confusenew.adjustSize()

        self.loadCsv(self.fileName)
        #----------------MESSAGE BOX-------------------------
        #self.generate.setEnabled(False)
        #self.getweet.setEnabled(False)
        final_results = final_report + "\n" +"------------------------------" + "\n" + "WITH HYBRID" + "\n" +  final_hyb_report
        self.result = QMessageBox()
        self.result.setWindowTitle("LOGISTIC REGRESSION")
        self.result.setText(final_results)
        self.result.setIcon(QMessageBox.Information)
        self.clear.setEnabled(True)
        self.generate.setEnabled(False)
        self.x = self.result.exec_()


    #-----------------FOR THE TABLE VIEW--------------------
    def loadCsv(self, fileName):
        with open(fileName, "r", encoding='utf8') as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)


    



























    def on_show(self,event):
        
        import BASELINE as twit #CODE FOR BASELINE 
        import re_log_Hyb as hyb #CODE FOR HYBRID

        final_hyb_accurate = str(hyb.percentage)
        final_hyb_cm  = str(hyb.confuse)
        final_hyb_report = str(hyb.reports)
        post_hyb_accurate = 'ACCURACY: ' +  final_hyb_accurate
        post_hyb_cm = 'CONFUSION MATRIX' + '\n' + final_hyb_cm 

        final_hybpos = str(hyb.posi)
        final_hybneu = str(hyb.neut)
        final_hybneg = str(hyb.nega)
        post_hybpos = 'POSITIVE: ' + final_hybpos
        post_hybneu = 'NEUTRAL: ' + final_hybneu
        post_hybneg = 'NEGATIVE: ' + final_hybneg
        post_hyboverall = 'OVERALL RESULT: ' + hyb.overall
        #-----PLOT GRAPH FOR OLD-----------------
        self.m.setHidden(False) 
        #----------------GRAPH FOR NEW-----------------------------
        self.n.setHidden(False) 

        #-----------------BASELINE RESULTS------------------------
        
        #self.vbox = QVBoxLayout()
        #self.canvas = twit.logicreg(self, width=8, height=6)
        #self.canvas.move(250, 100)
        final_accurate = str(twit.percentage)
        final_pos = str(twit.posi)
        final_neu = str(twit.neut)
        final_neg = str(twit.nega)
        final_cm  = str(twit.confuse)
        final_report = str(twit.reports)
        post_accurate = 'ACCURACY: ' +  final_accurate
        post_pos = 'POSITIVE: ' + final_pos
        post_neu = 'NEUTRAL: ' + final_neu
        post_neg = 'NEGATIVE: ' + final_neg
        post_overall = 'OVERALL RESULT: ' + twit.overall
        post_cm = 'CONFUSION MATRIX' + '\n' + final_cm
        #print(post_cm)






        #------------------TEXT CHANGED WITH THE RESULT----------------------------
       
        self.lbl_accuratenew.setText(post_hyb_accurate)
        self.lbl_accuratenew.adjustSize()

        self.lbl_accurateold.setText(post_accurate)
        self.lbl_accurateold.adjustSize()


        self.lbl_tally_positive.setText(post_pos)
        self.lbl_tally_positive.adjustSize()

        
        self.lbl_tally_neutral.setText(post_neu)
        self.lbl_tally_neutral.adjustSize()

        
        self.lbl_tally_negative.setText(post_neg)
        self.lbl_tally_negative.adjustSize()

        self.lbl_tally_overall.setText(post_overall)
        self.lbl_tally_overall.adjustSize()


        #-------------HYBRID---------------------
        self.lbl_tally_hybpositive.setText(post_hybpos)
        self.lbl_tally_hybpositive.adjustSize()

        
        self.lbl_tally_hybneutral.setText(post_hybneu)
        self.lbl_tally_hybneutral.adjustSize()

        
        self.lbl_tally_hybnegative.setText(post_hybneg)
        self.lbl_tally_hybnegative.adjustSize()

        self.lbl_tally_hyboverall.setText(post_hyboverall)
        self.lbl_tally_hyboverall.adjustSize()

        self.lbl_confuseold.setText(post_cm)
        self.lbl_confuseold.adjustSize()

        self.lbl_confusenew.setText(post_hyb_cm)
        self.lbl_confusenew.adjustSize()

        self.loadCsv(self.fileName)
        #----------------MESSAGE BOX-------------------------
        #self.generate.setEnabled(False)
        #self.getweet.setEnabled(False)
        final_results = final_report + "\n" +"------------------------------" + "\n" + "WITH HYBRID" + "\n" +  final_hyb_report
        self.result = QMessageBox()
        self.result.setWindowTitle("LOGISTIC REGRESSION")
        self.result.setText(final_results)
        self.result.setIcon(QMessageBox.Information)
        self.clear.setEnabled(True)
        self.generate.setHidden(False)
        self.generate.setEnabled(False)
        self.generatetwo.setHidden(True)
        self.x = self.result.exec_()
























    
    def terminate(self):
        exit()       





























    def on_clear(self):

        hide_seek = 1
        self.generate.setHidden(True)
        self.generatetwo.setHidden(False)
        self.generatetwo.setEnabled(True)
        #-----PLOT GRAPH FOR OLD-----------------
        self.m.setHidden(True) 
        #----------------GRAPH FOR NEW-----------------------------
        self.n.setHidden(True) 
        #------------------TEXT CHANGED WITH THE RESULT----------------------------
       
        self.lbl_accuratenew.setText('ACCURACY: ')
        self.lbl_accuratenew.adjustSize()

        self.lbl_accurateold.setText('ACCURACY: ')
        self.lbl_accurateold.adjustSize()


        self.lbl_tally_positive.setText('POSITIVE: ')
        self.lbl_tally_positive.adjustSize()

        
        self.lbl_tally_neutral.setText('NEUTRAL: ')
        self.lbl_tally_neutral.adjustSize()

        
        self.lbl_tally_negative.setText('NEGATIVE: ')
        self.lbl_tally_negative.adjustSize()

        self.lbl_tally_overall.setText('OVERALL RESULT: ')
        self.lbl_tally_overall.adjustSize()


        #-------------HYBRID---------------------
        self.lbl_tally_hybpositive.setText('POSITIVE: ')
        self.lbl_tally_hybpositive.adjustSize()

        
        self.lbl_tally_hybneutral.setText('NEUTRAL: ')
        self.lbl_tally_hybneutral.adjustSize()

        
        self.lbl_tally_hybnegative.setText('NEGATIVE: ')
        self.lbl_tally_hybnegative.adjustSize()

        self.lbl_tally_hyboverall.setText('OVERALL RESULT: ')
        self.lbl_tally_hyboverall.adjustSize()

        self.lbl_confuseold.setText('CONFUSION MATRIX')
        self.lbl_confuseold.adjustSize()

        self.lbl_confusenew.setText('CONFUSION MATRIX')
        self.lbl_confusenew.adjustSize()
        self.clear.setEnabled(False)
        self.generate.setEnabled(True)



    

















    



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = MyFirstGUI()
    gui.show()
    sys.exit(app.exec_())