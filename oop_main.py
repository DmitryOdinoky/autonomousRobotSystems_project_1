import PySimpleGUI as sg
import numpy as np
import math
# import random
import os

my_absolute_dirpath = os.getcwd()
my_absolute_dirpath.replace('\\' ,'/')
my_absolute_dirpath = my_absolute_dirpath.replace('\\' ,'/')


sg.theme('DarkGrey5')
_VARS = {'cellCount': 10, 'gridSize': 400, 'canvas': False, 'window': False,
         'playerPos': [40*5, 40*3, 0], 'cellMAP': False, 'cellMapTags': False, 
         'measuredMatrix': False, 'groundTruthMatrix': False, 'weightsMatrix': False, 'odoWeightsMatrix': False,
         'sensorsError': False, 'odoError': False}


class myGame:

    def __init__(self, _VARS):
        
        self._VARS = _VARS
        
        self.uploadMaze('/autonomousRobotSystems_project_1/assets/worldMaps/WM_10_x_10.csv')
        
        self.dict = {
            
            0: 'Empty_UP',
            # 1 is for obstacle and is skipped
            1: 'Empty_DOWN',
            2: 'Empty_LEFT',
            3: 'Empty_RIGHT',            
            
            4: 'Corner_UP',
            5: 'Corner_DOWN',
            6: 'Corner_LEFT',
            7: 'Corner_RIGHT',
            
            8: 'Corridor_UP',
            9: 'Corridor_DOWN',
            10: 'Corridor_LEFT',
            11: 'Corridor_RIGHT',
            
            12: 'Deadlock_UP',
            13: 'Deadlock_DOWN',
            14: 'Deadlock_LEFT',
            15: 'Deadlock_RIGHT',     
            
            16: 'Wall_UP',
            17: 'Wall_DOWN',
            18: 'Wall_LEFT',
            19: 'Wall_RIGHT'               

            }

        
        
        self.cellSize = self._VARS['gridSize']/self._VARS['cellCount']
        self.exitPos = [self._VARS['cellCount']-1, self._VARS['cellCount']-1]
        self.celCount = self._VARS['cellCount']
        self.starterMap = np.zeros((self.celCount, self.celCount), dtype=int)
        self.tagsMap = np.zeros((self.celCount, self.celCount), dtype=int)
        
        # self.currentPosition = None
        
        self.probability = None
        self.markovWeights = None
        
        self.sensorsError = None
        self.odoError = None
        
        self.reshapedMap = None
        
        self.generatedMeasureMatrix = None
        self.generatedWeightsMatrix = None
        self.odoWeightsMatrix = None
        self.groundTruthMatrix = None
        
        self.AppFont = 'Any 16'
        self.cellSize = _VARS['gridSize']/_VARS['cellCount']
        self.exitPos = [_VARS['cellCount']-1, _VARS['cellCount']-1]
        
        self.putTagsToMap('/autonomousRobotSystems_project_1/assets/worldMaps/WM_10_x_10_tags.csv')
        self.uploadModels('/autonomousRobotSystems_project_1/assets/models/sensors_error_model.csv', '/autonomousRobotSystems_project_1/assets/models/movement_error_model.csv')
        self.initMarkovAlgo()
        

        
        self.uploadModels(sensorsError = '/autonomousRobotSystems_project_1/assets/models/sensors_error_model.csv', odoError = '/autonomousRobotSystems_project_1/assets/models/movement_error_model.csv')
        
        self.tagToMeasurement()
        
        print('sds')
        


    def uploadMaze(self, rel_path):


        self.starterMap = np.genfromtxt(my_absolute_dirpath + rel_path, delimiter=',')
        self._VARS['cellMAP'] = self.starterMap

        self._VARS['cellCount'] = self._VARS['cellMAP'].shape[0]
        # print (starterMap)
        return self.starterMap    

    def putTagsToMap(self, rel_path):

        self.tagsMap = np.genfromtxt(my_absolute_dirpath + rel_path, delimiter=',')
        self._VARS['cellMapTags'] = self.tagsMap
        
        return self.tagsMap 
        

    def uploadModels(self, sensorsError, odoError):
         
        self.sensorsError = np.genfromtxt(my_absolute_dirpath + sensorsError, delimiter=',')
        self._VARS['sensorsError'] = self.sensorsError
            
        self.odoError = np.genfromtxt(my_absolute_dirpath + odoError, delimiter=',')
        self._VARS['odoError'] = self.odoError
        
        return self.sensorsError, self.odoError
    
    def initMarkovAlgo(self):
       
        self.reshapedMap = np.reshape(self._VARS['cellMapTags'], self._VARS['cellCount']**2)
        self.probability = np.copy(self.reshapedMap[::])
        self.probability[:] = 0.1
        
        self.markovWeights = self.probability[::]
        self.odoWeightsMatrix = self.probability[::]
        self._VARS['weightsMatrix'] = np.reshape(self.markovWeights, (self._VARS['cellCount'], self._VARS['cellCount']))
        self._VARS['odoWeightsMatrix'] = np.reshape(self.odoWeightsMatrix, (self._VARS['cellCount'], self._VARS['cellCount']))
    
    def tagToMeasurement(self):
        
        self.generatedMeasureMatrix = []
        self.groundTruthMatrix = []
        
        # Possible measurements for every cell are generated, based on error model. 
        # Each configuration is combined with one of 4 'theta' angles / direcrions. 5 configurations * 4 directions = 20 possible configurations
        
        # 0 - empty, no walls nearby
        # 1 - obstacle
        # 2 - corner
        # 3 - corridor
        # 4 - deadlock
        # 5 - single wall
        
        for cell in self.reshapedMap:
            
            if (cell == 1): 

                cell = 20
                trueCell = 20
                
            elif (cell == 0 and self._VARS['playerPos'][2] == 90):
                

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[0,::].tolist())
                trueCell = 0

            elif (cell == 0 and self._VARS['playerPos'][2] == 270):
                

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[1,::].tolist())  
                trueCell = 1

            elif (cell == 0 and self._VARS['playerPos'][2] == 180):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[2,::].tolist())
                trueCell = 2

            elif (cell == 0 and self._VARS['playerPos'][2] == 0):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[3,::].tolist())
                trueCell = 3
                
#===

            elif (cell == 2 and self._VARS['playerPos'][2] == 90):
                
                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[4,::].tolist())
                trueCell = 4

            elif (cell == 2 and self._VARS['playerPos'][2] == 270):
                
                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[5,::].tolist())
                trueCell = 5

            elif (cell == 2 and self._VARS['playerPos'][2] == 180):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[6,::].tolist())
                trueCell = 6

            elif (cell == 2 and self._VARS['playerPos'][2] == 0):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[7,::].tolist())
                trueCell = 7
                 

#===

            elif (cell == 3 and self._VARS['playerPos'][2] == 90):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[8,::].tolist())
                trueCell = 8

            elif (cell == 3 and self._VARS['playerPos'][2] == 270):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[9,::].tolist())
                trueCell = 9

            elif (cell == 3 and self._VARS['playerPos'][2] == 180):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[10,::].tolist())
                trueCell = 10

            elif (cell == 3 and self._VARS['playerPos'][2] == 0):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[11,::].tolist())
                trueCell = 11
                
#===

            elif (cell == 4 and self._VARS['playerPos'][2] == 90):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[12,::].tolist())
                trueCell = 12

            elif (cell == 4 and self._VARS['playerPos'][2] == 270):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[13,::].tolist())
                trueCell = 13

            elif (cell == 4 and self._VARS['playerPos'][2] == 180):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[14,::].tolist())
                trueCell = 14

            elif (cell == 4 and self._VARS['playerPos'][2] == 0):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[15,::].tolist())
                trueCell = 15

#===

            elif (cell == 5 and self._VARS['playerPos'][2] == 90):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[16,::].tolist())
                trueCell = 16

            elif (cell == 5 and self._VARS['playerPos'][2] == 270):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[17,::].tolist())
                trueCell = 17

            elif (cell == 5 and self._VARS['playerPos'][2] == 180):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[18,::].tolist())
                trueCell = 18

            elif (cell == 5 and self._VARS['playerPos'][2] == 0):

                cell = np.random.choice([*self.dict], replace=True, p=self.sensorsError[19,::].tolist())
                trueCell = 19

            self.generatedMeasureMatrix.append(cell)
            self.groundTruthMatrix.append(trueCell)

        self.generatedMeasureMatrix = np.array(self.generatedMeasureMatrix)
        self.groundTruthMatrix = np.array(self.groundTruthMatrix)               
                
        self._VARS['measuredMatrix'] = np.reshape(self.generatedMeasureMatrix, (self._VARS['cellCount'], self._VARS['cellCount']))
        self._VARS['groundTruthMatrix'] = np.reshape(self.groundTruthMatrix, (self._VARS['cellCount'], self._VARS['cellCount']))
        
    def iterateMarkovAlgoMeasure(self):
        
            # xPos = int(math.ceil(self._VARS['playerPos'][0]/self.cellSize))
            # yPos = int(math.ceil(self._VARS['playerPos'][1]/self.cellSize))
            # theta = self._VARS['playerPos'][2]
            
            # measuredCell = self._VARS['measuredMatrix'][yPos][xPos]
            
            generatedWeightsMatrix = []
            
            for cell, trueCell, weight in zip(self.generatedMeasureMatrix, self.groundTruthMatrix, self.markovWeights):
                
                
                if(cell==20):
                    
                    out = 20
                
                elif(cell==trueCell and cell!=20):
                
                    out = weight*self.sensorsError[cell, cell]
                    
                else:
                    
                    out = weight*self.sensorsError[cell, trueCell]
                    
                generatedWeightsMatrix.append(out)
                    
                
            self.generatedWeightsMatrix = np.array(generatedWeightsMatrix)
            
            arrSum = []
            maximum = np.max(self.generatedWeightsMatrix)
            for cell in self.generatedWeightsMatrix:
                
                if(cell!=maximum): arrSum.append(cell)
                
            arrSum = np.array(arrSum)
            
            mySum = np.sum(arrSum)
            
            
            self.generatedWeightsMatrix = self.generatedWeightsMatrix/mySum
            
            self._VARS['weightsMatrix'] = np.reshape(self.generatedWeightsMatrix, (self._VARS['cellCount'], self._VARS['cellCount']))
                    
                    
            
            # if(self._VARS['measuredMatrix'][yPos][xPos] == 0 and theta==0): 
                
    def iterateMarkovAlgoMove(self):

        odoError = np.random.choice([self.odoError[0],self.odoError[1]], replace=True, p=self.odoError[::].tolist()) 

        
        odoArr = []                       
        for cell in self.generatedWeightsMatrix:
            out = cell*odoError
            odoArr.append(out)
        self.odoWeightsMatrix = np.array(odoArr)
        self._VARS['odoWeightsMatrix'] = np.reshape(self.odoWeightsMatrix, (self._VARS['cellCount'], self._VARS['cellCount']))
        
        
            
            
            
    
    def drawGrid(self):
        cells = self._VARS['cellCount']
        self._VARS['canvas'].TKCanvas.create_rectangle(
            1, 1, self._VARS['gridSize'], self._VARS['gridSize'], outline='BLACK', width=1)
        for x in range(cells):
            self._VARS['canvas'].TKCanvas.create_line(
                ((self.cellSize * x), 0), ((self.cellSize * x), self._VARS['gridSize']),
                fill='BLACK', width=1)
            self._VARS['canvas'].TKCanvas.create_line(
                (0, (self.cellSize * x)), (self._VARS['gridSize'], (self.cellSize * x)),
                fill='BLACK', width=1)
            
        
    def drawCell(self, x, y, color='GREY'):
        self._VARS['canvas'].TKCanvas.create_rectangle(
            x, y, x + self.cellSize, y + self.cellSize,
            outline='BLACK', fill=color, width=1)
    
    
    def placeCells(self):
        for row in range(self._VARS['cellMAP'].shape[0]):
            for column in range(self._VARS['cellMAP'].shape[1]):
                if(self._VARS['cellMAP'][column][row] == 1):
                    self.drawCell((self.cellSize*row), (self.cellSize*column))
    
    def saveWeights(self, idx):
        
        np.savetxt(f'measure_step_{idx}.csv', self._VARS['weightsMatrix'], delimiter=",")
        np.savetxt(f'odo_step_{idx}.csv', self._VARS['odoWeightsMatrix'], delimiter=",")

    def checkEvents(self, event):
        move = ''
        if len(event) == 1:
            if ord(event) == 63232:  # UP
                move = 'Up'
            elif ord(event) == 63233:  # DOWN
                move = 'Down'
            elif ord(event) == 63234:  # LEFT
                move = 'Left'
            elif ord(event) == 63235:  # RIGHT
                move = 'Right'
        # Filter key press Windows :
        else:
            if event.startswith('Up'):
                move = 'Up'
            elif event.startswith('Down'):
                move = 'Down'
            elif event.startswith('Left'):
                move = 'Left'
            elif event.startswith('Right'):
                move = 'Right'
        return move
    
    def checkObjects(self, xPos, yPos):
        
            if self._VARS['cellMAP'][yPos-1][xPos] == 1:
                print(f'Object found to the UP (x:{[xPos]}), (y:{yPos-1})')
            if self._VARS['cellMAP'][yPos+1][xPos] == 1:
                print(f'Object found to the DOWN (x:{[xPos]}), (y:{yPos+1})')
            if self._VARS['cellMAP'][yPos][xPos-1] == 1:
                print(f'Object found to the LEFT  (x:{[xPos-1]}), (y:{yPos})')
            if self._VARS['cellMAP'][yPos][xPos+1] == 1:
                print(f'Object found to the RIGHT (x:{[xPos+1]}), (y:{yPos})')  
                
            print('=================')

    def initGame(self):
        
        layout = [[sg.Canvas(size=(self._VARS['gridSize'], self._VARS['gridSize']),
                     background_color='white',
                     key='canvas')],
                  [sg.Exit(font=self.AppFont),
           sg.Text('', key='-exit-', font=self.AppFont, size=(15, 1)),
           sg.Button('Save_Weights', font=self.AppFont)
           ]]
        
        self._VARS['window'] = sg.Window('World', layout, resizable=True, finalize=True,
                            return_keyboard_events=True)
        self._VARS['canvas'] = self._VARS['window']['canvas']
        self.drawGrid()
        self.drawCell(_VARS['playerPos'][0], _VARS['playerPos'][1], 'TOMATO')
        # drawCell(_VARS['playerPos_2'][0], _VARS['playerPos_2'][1], 'TOMATO')
        # drawCell(_VARS['playerPos_3'][0], _VARS['playerPos_3'][1], 'TOMATO')
        # self.drawCell(exitPos[0]*cellSize, exitPos[1]*cellSize, 'White')
        # self.placeCells()
        
        idx = 0
        
        while True:             # Event Loop

            # Clear canvas, draw grid and cells
            self._VARS['canvas'].TKCanvas.delete("all")
            self.drawGrid()
            
            

                
                


            xPos = int(math.ceil(self._VARS['playerPos'][0]/self.cellSize))
            yPos = int(math.ceil(self._VARS['playerPos'][1]/self.cellSize))
            theta = self._VARS['playerPos'][2]
            
            

            self.checkObjects(xPos, yPos)
            self.iterateMarkovAlgoMeasure()
                         
            

            
            step = self._VARS['gridSize']/self._VARS['cellCount']
            
            self.drawCell(_VARS['playerPos'][0], _VARS['playerPos'][1], 'TOMATO')
            
            self.drawCell(_VARS['playerPos'][0]+step, _VARS['playerPos'][1], 'GREEN')
            self.drawCell(_VARS['playerPos'][0]-step, _VARS['playerPos'][1], 'GREEN')
            self.drawCell(_VARS['playerPos'][0], _VARS['playerPos'][1]+step, 'GREEN')
            self.drawCell(_VARS['playerPos'][0], _VARS['playerPos'][1]-step, 'GREEN')

            
            self.placeCells()                             
   

            event, values = self._VARS['window'].read()
            
         

        
            
            # Filter key press
  
            self.tagToMeasurement()
            odoError = np.random.choice([1,0], replace=True, p=self.odoError[::].tolist())
                
            if (self.checkEvents(event) == 'Up' and odoError == 1):
                self._VARS['playerPos'][2] = 90
                
                self.iterateMarkovAlgoMove()             
                if int(self._VARS['playerPos'][1] - self.cellSize) >= 0:
                    
                    
                    if self._VARS['cellMAP'][yPos-1][xPos] != 1:
                        
                        self._VARS['playerPos'][1] = self._VARS['playerPos'][1] - self.cellSize
                    else: 
                        print('ILLEGAL MOVE!')                        
                        
                
                # self.checkObjects(xPos, yPos)
                        

            elif (self.checkEvents(event) == 'Down' and odoError == 1):
                self._VARS['playerPos'][2] = 270
                self.iterateMarkovAlgoMove()
                
                if int(self._VARS['playerPos'][1] + self.cellSize) < self._VARS['gridSize']-1:
                    
                    
                    if self._VARS['cellMAP'][yPos+1][xPos] != 1:
                        self._VARS['playerPos'][1] = self._VARS['playerPos'][1] + self.cellSize
                    else: 
                        print('ILLEGAL MOVE!')                        
                        
                # self.checkObjects(xPos, yPos)
                        
                
            elif (self.checkEvents(event) == 'Left' and odoError == 1):
                self._VARS['playerPos'][2] = 180
                self.iterateMarkovAlgoMove()
                
                
                if int(self._VARS['playerPos'][0] - self.cellSize) >= 0:
                    
                    if self._VARS['cellMAP'][yPos][xPos-1] != 1:
                        self._VARS['playerPos'][0] = self._VARS['playerPos'][0] - self.cellSize
                    else: 
                        print('ILLEGAL MOVE!')                        
                        
                # self.checkObjects(xPos, yPos)        
                        
            elif (self.checkEvents(event) == 'Right' and odoError == 1):
                self._VARS['playerPos'][2] = 0
                self.iterateMarkovAlgoMove()
                
                
                if int(self._VARS['playerPos'][0] + self.cellSize) < self._VARS['gridSize']-1:
                    
                    if self._VARS['cellMAP'][yPos][xPos+1] != 1:
                        self._VARS['playerPos'][0] = self._VARS['playerPos'][0] + self.cellSize
                    else: 
                        print('ILLEGAL MOVE!')
            else:
                
                print('ROBOT FAILED TO MOVE!')
                        
            
                
                # self.checkObjects(xPos, yPos)

            xPos = int(math.ceil(self._VARS['playerPos'][0]/self.cellSize))
            yPos = int(math.ceil(self._VARS['playerPos'][1]/self.cellSize))
            theta = self._VARS['playerPos'][2]                        
        
            print(f'Actual position: (x:{[xPos]}), (y:{[yPos]}), (theta: {theta} degrees)')
            
            
            
            if event == 'Save_Weights':
                
                self.saveWeights(idx)   
                
            idx+=1


            


            
            self._VARS['window']['-exit-'].update('')

            if event in (None, 'Exit'):
                break
        
        self._VARS['window'].close()        

                            

newGame = myGame(_VARS)


newGame.initGame()






