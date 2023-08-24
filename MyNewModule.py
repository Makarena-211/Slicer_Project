import qt
import slicer
import ctk
import vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from sklearn.preprocessing import MinMaxScaler
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import json
import requests
from vtk.util import numpy_support

class MyNewModule:
    def __init__(self, parent):  # функция с названиями, инициалами и тд
        parent.title = "My Module"
        parent.categories = ["My category"]
        parent.dependencies = []
        parent.contributors = ["mnfomenkov@gmail.com"]
        parent.helpText = "Example of loading DICOM files and creating its mask"
        parent.acknowledgementText = "This file was originally developed by Makarena"
        self.parent = parent

class MyNewModuleWidget:
    def __init__(self, parent=None):
        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
        self.layout = self.parent.layout()
        if not parent:
            self.setup()
            self.parent.show()

    def setup(self):
        collapsibleButton = ctk.ctkCollapsibleButton() #кнопка меню
        collapsibleButton.text = "MyCollapsibleMenu" #название кнопки

        self.layout.addWidget(collapsibleButton)
        self.formLayout = qt.QFormLayout(collapsibleButton)
        self.formFrame = qt.QFrame(collapsibleButton)
        self.formFrame.setLayout(qt.QHBoxLayout())
        self.formLayout.addRow(self.formFrame)

        self.inputSelectorLabel = qt.QLabel("Input Volume:", self.formFrame)
        self.formFrame.layout().addWidget(self.inputSelectorLabel)

        self.inputSelector = slicer.qMRMLNodeComboBox(self.formFrame)
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLVectorVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.formFrame.layout().addWidget(self.inputSelector)

        button = qt.QPushButton("Get RGB array")
        button.toolTip = "Displays the path of the selected volume"
        button.connect("clicked(bool)", self.pixelArray)
        self.formFrame.layout().addWidget(button)

        button2 = qt.QPushButton("Markupssss")
        button2.connect("clicked(bool)", self.get_many_coords)
        self.formFrame.layout().addWidget(button2)

        button3 = qt.QPushButton("Roi")
        button3.connect("clicked(bool)", self.get_roi)
        self.formFrame.layout().addWidget(button3)

        button4 = qt.QPushButton("Load to json")
        button4.connect("clicked(bool)", self.to_JSON)
        self.formFrame.layout().addWidget(button4)

        button5 = qt.QPushButton("Scan")
        button5.connect("clicked(bool)", self.scan)
        self.formFrame.layout().addWidget(button5)

        button6 = qt.QPushButton("Mask")
        button6.connect("clicked(bool)", self.creating_mask)
        self.formFrame.layout().addWidget(button6)

        self.textfield = qt.QTextEdit()  # текстовое поле только для чтения
        self.textfield.setReadOnly(True)
        self.formFrame.layout().addWidget(self.textfield)

    def informationButtonClicked(self):
        # Получение пути выбранного узла
        currentNode = self.inputSelector.currentNode()
        if currentNode:
            storageNode = currentNode.GetStorageNode()
            if storageNode:
                path = storageNode.GetFileName()
                self.textfield.insertPlainText(path)
            else:
                self.textfield.insertPlainText("No storage node associated with the selected node")
        else:
            self.textfield.insertPlainText("No node selected")



    def get_many_coords(self): #пустой если обернуть в list()
        fiducial_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")  # список всех узлов, которые точки
        coordinates = []  # полготавливаем список для его координат
        for node in fiducial_nodes:  # итерируемся по каждому узлу
            fiducial_name = node.GetName() #получаем имя узла
            for i in range(node.GetNumberOfControlPoints()):   #каждая точка из списка
                control_point = node.GetNthControlPointPositionWorld(i)  #получаем координаты
                coordinates.append(control_point) #добавляем координаты


        print("Coordinates:")
        for i, coord in enumerate(coordinates, 1):
            print(f"Point {i}: {coord}")
        print(coordinates)
        return coordinates


    def get_roi(self):
        roi_nodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode') #получили список узлов которые области
        all_roi_coords = []
        for roi_node in roi_nodes: #roi_nodes = пустой массив, если roi нет на view  port
            a = roi_node.GetName() #получаем имя каждого узла
            roi = slicer.util.getNode(a) #получаем всю инфу по каждому roi
            center = roi.GetNthControlPointPositionWorld(0)
            size = roi.GetSize()
            size2d = [size[0]/2, size[1]/2] #половина каждой стороны

            x_left_top = center[0] - size2d[0]
            y_left_top = center[1] + size2d[1]
            x_right_bottom = center[0] + size2d[0]
            y_right_bottom = center[1] - size2d[1]
            roi_coords = [x_left_top, y_left_top, x_right_bottom, y_right_bottom]

            all_roi_coords.append(roi_coords)
            print(f'Координаты лево вверх и права низ = {roi_coords}') # эти коорды мы будем закидывать в sam
        print(all_roi_coords)
        return all_roi_coords


    def pixelArray(self):
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") #если пустой то = None
        pixel_array = slicer.util.arrayFromVolume(mainvolume) #получили pixel_array изображения
        return pixel_array

    def scan(self):
        fiducial_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode") #точки
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") #фотка
        roi_nodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode') #область
        if len(list(fiducial_nodes)) == 0:
            print('No points')
        if len(list(fiducial_nodes)) != 0:
            self.get_many_coords()
        if len(roi_nodes) == 0:
            print('No ROI')
        if len(roi_nodes) != 0:
            self.get_roi()
        if mainvolume is None:
            print('No photo')
        if mainvolume:
            self.pixelArray()

    def to_JSON(self):
        outputFileName = r"C:\Users\mnfom\Documents\Работа\ML\pythonProject\data.json"
        url = 'http://127.0.0.1:8000/masks'

        try:
            points = self.get_many_coords()
            points_serializable = [list(coord) for coord in points]  # Преобразовываем vtkVector3d в список
        except Exception as e:
            points_serializable = []

        try:
            roi = self.get_roi()
        except Exception as e:
            roi = []

        try:
            pixel_arr = self.pixelArray()
            pixel_arr_serializable = pixel_arr.tolist()  # Преобразовываем ndarray в список
        except Exception as e:
            pixel_arr_serializable = []

        data = {
            "points": points_serializable,  # Используем преобразованные данные
            "roi": roi,
            'pixel_arr': pixel_arr_serializable
        }


        #print(len(data["pixel_arr"]))
        #print(type(points_serializable), type(roi), type(pixel_arr_serializable))
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            print("File sent successfully to the server.")
            data_mask = response.content.decode()
        else:
            print(f"Failed to send file. Response status code: {response.status_code}")
            data_mask = {}


        #print(data_mask)
        return data_mask

    def creating_mask(self):
        mask_data = self.to_JSON()
        mask = json.loads(mask_data)
        mask_points = np.array(mask.get("mask_points", []))
        mask_roi = np.array(mask.get("mask_roi", []))

        #пометка: мб стоит отказаться от этого решения в сторону отображения по цветам церез цикл
        combined_mask = np.logical_or(mask_points, mask_roi)

        #print(combined_mask)  # после этого идут преобразования






''' 
        with open(outputFileName, 'w') as outfile:
            json.dump(data, outfile)

        with open('data.json', 'rb') as file:
            files = {'file': ('data.json', file, 'application/json')}
            response = requests.post(url, headers=headers, files=files)
'''



#ошибка: ValueError: dictionary update sequence element #0 has length 320; 2 is required
#←[32mINFO←[0m:     127.0.0.1:55334 - "←[1mGET /masks HTTP/1.1←[0m" ←[31m422 Unprocessable Entity←[0m