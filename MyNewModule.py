import qt
import slicer
import ctk
import vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import getNode
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

class SammBaseLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()
        self._connections = None
        self._flag_prompt_sync = False
        self._flag_promptpts_sync = False
        self._frozenSlice = {"R": [], "G": [], "Y": []}
        #self._latlogger = LatencyLogger()

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

        button7 = qt.QPushButton("RAS")
        button7.connect("clicked(bool)", self.ras_to_ijk)
        self.formFrame.layout().addWidget(button7)


        self.textfield = qt.QTextEdit()  # текстовое поле только для чтения
        self.textfield.setReadOnly(True)
        self.textfield.setReadOnly(True)
        self.formFrame.layout().addWidget(self.textfield)


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

    def ras_to_ijk(self):
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        ras = self.get_many_coords()
        ras_serializable = [list(coord) for coord in ras]
        ras_serializable = [item for sublist in ras_serializable for item in sublist]
        print(f"От функции: {type(ras)}")
        print(f"Сериализация {ras_serializable}")
        rasToIjkMatrix = vtk.vtkMatrix4x4()
        mainvolume.GetRASToIJKMatrix(rasToIjkMatrix)
        ras_serializable.append(1)

        ijk = [0, 0, 0, 0]
        for i in range(4):
            for j in range(4):
                ijk[i] += rasToIjkMatrix.GetElement(i, j) * ras_serializable[j]

        ijk.pop()
        ijk.pop()
        # округляем значения до целых чисел
        #ijk = [int(round(x)) for x in ijk]
        print(f"Результат {ijk}, {type(ijk)}")
        return ijk

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
            mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            ras = self.get_many_coords()
            ras_serializable = [list(coord) for coord in ras]
            ras_serializable = [item for sublist in ras_serializable for item in sublist]
            print(f"От функции: {type(ras)}")
            print(f"Сериализация {ras_serializable}")
            rasToIjkMatrix = vtk.vtkMatrix4x4()
            mainvolume.GetRASToIJKMatrix(rasToIjkMatrix)
            ras_serializable.append(1)

            ijk = [0, 0, 0, 0]
            for i in range(4):
                for j in range(4):
                    ijk[i] += rasToIjkMatrix.GetElement(i, j) * ras_serializable[j]

            ijk.pop()
            ijk.pop()
            # округляем значения до целых чисел
            # ijk = [int(round(x)) for x in ijk]
            print(f"Результат {ijk}, {type(ijk)}")
        except Exception as e:
            ijk = []

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
            "points": ijk,  # Используем преобразованные данные
            "roi": roi,
            'pixel_arr': pixel_arr_serializable
        }

        print(f'Внутренность JSON: {data["roi"]}, {data["points"]}')
        print(len(data))


        #print(len(data["pixel_arr"]))
        #print(type(points_serializable), type(roi), type(pixel_arr_serializable))
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            print("File sent successfully to the server.")
            data_mask = response.content.decode()
            #print(data_mask)
            return data_mask
        else:
            print(f"Failed to send file. Response status code: {response.status_code}")
            data_mask = []
            return data_mask


    def creating_mask(self):
        mask_data = self.to_JSON()  #строка

        mask = json.loads(mask_data) #словарь с масками
        mask_points = np.array(mask["mask_points"])
        mask_roi = np.array(mask["mask_roi"])
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        arr_volume = slicer.util.arrayFromVolume(mainvolume)
        labelmap_data = mask_points.astype(int)
        mask3d = labelmap_data[0,:,:]
        print(mask_points, mask_points.shape)
        print(mask3d, mask3d.shape)

        point_Ras = [0, 0, 0, 1]  #начальная точка RAS
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, mainvolume.GetParentTransformNode(),
                                                             transformRasToVolumeRas)
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(point_Ras[0:3])

        volumeRasToIjk = vtk.vtkMatrix4x4()
        mainvolume.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
        point_Ijk = [c for c in point_Ijk[0:3]]

        T = np.array(
            [[-1, 0, 0, point_Ijk[0]],
             [0, -1, 0, point_Ijk[1]],
             [0, 0, 1, -point_Ijk[2]],
             [0, 0, 0, 1]])

        volumeNode = slicer.util.addVolumeFromArray(mask3d * 120, T, nodeClassName="vtkMRMLLabelMapVolumeNode")

        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", 'Segmentation_prostate')
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(volumeNode, seg)
        seg.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(volumeNode)
        segmentationNode = getNode('Segmentation_prostate')
        segmentation = segmentationNode.GetSegmentation()
        segment = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
        segment.SetColor(230 / 255.0, 158 / 255.0, 140 / 255.0)
        segment.SetName('prostate')







''' 
        with open(outputFileName, 'w') as outfile:
            json.dump(data, outfile)

        with open('data.json', 'rb') as file:
            files = {'file': ('data.json', file, 'application/json')}
            response = requests.post(url, headers=headers, files=files)
'''



#ошибка: ValueError: dictionary update sequence element #0 has length 320; 2 is required
#←[32mINFO←[0m:     127.0.0.1:55334 - "←[1mGET /masks HTTP/1.1←[0m" ←[31m422 Unprocessable Entity←[0m
#r"C:\Users\mnfom\Documents\Работа\ML\pythonProject\mask.json"
