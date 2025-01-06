import qt
import slicer
import ctk
import vtk
from slicer.ScriptedLoadableModule import *
import numpy as np
import json
import requests
#from vtk.util import numpy_support
#from slicer.util import getNode
#from sklearn.preprocessing import MinMaxScaler
#import torch
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class MyNewModule:
    def __init__(self, parent):  # функция с названиями, инициалами и тд
        parent.title = "MLMedicine Module"
        parent.categories = ["SAMSegment"]
        parent.dependencies = []
        parent.contributors = ["mnfomenkov@gmail.com, tg: @heeeey_makarena"]
        parent.helpText = "Place point or roi to create a mask on your DICOM"
        parent.acknowledgementText = "This file was originally developed by Makar Fomenkov"
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

        #self.inputSelectorLabel = qt.QLabel("Input Volume:", self.formFrame)
        #self.formFrame.layout().addWidget(self.inputSelectorLabel)

        # self.inputSelector = slicer.qMRMLNodeComboBox(self.formFrame)
        # self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLVectorVolumeNode"]
        # self.inputSelector.addEnabled = False
        # self.inputSelector.removeEnabled = False
        # self.inputSelector.setMRMLScene(slicer.mrmlScene)
        # self.formFrame.layout().addWidget(self.inputSelector)
        button4 = qt.QPushButton("Slice")
        button4.connect("clicked(bool)", self.current_slice)
        self.formFrame.layout().addWidget(button4)

        button5 = qt.QPushButton("Scan")
        button5.connect("clicked(bool)", self.scan)
        self.formFrame.layout().addWidget(button5)

        button6 = qt.QPushButton("Mask")
        button6.connect("clicked(bool)", self.creating_mask)
        self.formFrame.layout().addWidget(button6)

        button1 = qt.QPushButton("shape")
        button1.connect("clicked(bool)", self.pixelArray)
        self.formFrame.layout().addWidget(button1)

        checkbox1 = qt.QCheckBox("Positive")
        self.formFrame.layout().addWidget(checkbox1)
        # Присваиваем функцию обработчика события
        checkbox1.stateChanged.connect(self.checkbox1_red)

    def input_label(self):
        input_label = []
        fiducialNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        for fiducialNode in fiducialNodes:
            mDisplayNode = fiducialNode.GetDisplayNode()
            point_color = mDisplayNode.GetColor()
            if point_color == (0, 1, 0):
                input_label.append(1)
            else:
                input_label.append(0)
        print(input_label)
        return input_label

    def func1(self):
        layoutManager = slicer.app.layoutManager()
        red = layoutManager.sliceWidget("Red")
        redLogic = red.sliceLogic()
        print(redLogic.GetSliceOffset())


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



    def checkbox1_red(self, state):
        input_label = []
        if state == qt.Qt.Checked:
            print("Positive")
            fiducialNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
            if len(fiducialNodes) > 0:
                lastFiducialNode = fiducialNodes[-1]
                mDisplayNode = lastFiducialNode.GetDisplayNode()
                mDisplayNode.SetColor(0, 1, 0)
                mDisplayNode.SetSelectedColor(0, 1, 0)

        else:
            print("Negative")
            fiducialNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
            if len(fiducialNodes) > 0:
                lastFiducialNode = fiducialNodes[-1]
                mDisplayNode = lastFiducialNode.GetDisplayNode()
                mDisplayNode.SetColor(255, 0, 0)
                mDisplayNode.SetSelectedColor(255, 0, 0)


    def get_roi(self): #центр и позиция одно и тоже
        roi_nodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        ras_roi_all = []
        sizes = []
        for roi_node in roi_nodes:
            sizes.append(roi_node.GetSize())
            sizes = [list(size) for size in sizes]
            for i in range(0, roi_node.GetNumberOfControlPoints()):
                centers = roi_node.GetNthControlPointPositionWorld(i)
                ras_roi_all.append(centers)
                ras_roi_all = [list(coord) for coord in ras_roi_all]
        print(f"центр: {ras_roi_all}, размер: {sizes}")
        return ras_roi_all, sizes


    def pixelArray(self):
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode") #если пустой то = None
        pixel_array = slicer.util.arrayFromVolume(mainvolume) #получили pixel_array изображения
        print(pixel_array.shape)
        return pixel_array


    def ras_to_ijk(self, coords_list):
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        rasToIjkMatrix = vtk.vtkMatrix4x4()
        mainvolume.GetRASToIJKMatrix(rasToIjkMatrix)
        coords_list.append(1)
        ijk = [0, 0, 0, 0]
        for i in range(4): #циклы для прохода по матрице и по списку коорд
            for j in range(4):
                ijk[i] += rasToIjkMatrix.GetElement(i, j) * coords_list[j]
        ijk.pop()
        ijk.pop()
        print(f"Результат {ijk}, {type(ijk)}")
        return ijk

    def ras_to_ijk_roi(self):
        ijk_coords=[]
        mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        ras_roi_all, sizes = self.get_roi()
        for i in range(len(ras_roi_all)):
            for j in range(len(sizes)):
                ras_roi = ras_roi_all[i]
                size = sizes[j]
            if ras_roi:
                print(f"Коорды roi из функции {ras_roi}, {type(ras_roi)}, {size}")
                size2d = [size[0] / 2, size[1] / 2]
                x_left_top = ras_roi[0] - size2d[0]
                y_left_top = ras_roi[1] + size2d[1]
                x_right_bottom = ras_roi[0] + size2d[0]
                y_right_bottom = ras_roi[1] - size2d[1]
                ras_coords_left = [x_left_top, y_left_top, ras_roi[2]]
                ras_coords_right = [x_right_bottom, y_right_bottom, ras_roi[2]]
                ijk_coords.append(self.ras_to_ijk(ras_coords_left) + self.ras_to_ijk(ras_coords_right))
                for b in range(len(ijk_coords)):
                    ijk_coords[b][0], ijk_coords[b][1], ijk_coords[b][2], ijk_coords[b][3] = ijk_coords[b][2], ijk_coords[b][1], ijk_coords[b][0], ijk_coords[b][3]
        print(f"результат {ijk_coords}")
        return ijk_coords



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

    def current_slice(self):
        xyz = [0.0, 0.0, 0.0]
        layoutManager = slicer.app.layoutManager()
        red = layoutManager.sliceWidget("Red")
        redLogic = red.sliceLogic()
        layerLogic = redLogic.GetBackgroundLayer()
        xyToIJK = layerLogic.GetXYToIJKTransform()
        ijkFloat = xyToIJK.TransformDoublePoint(xyz)
        print(f"current slice: {int(ijkFloat[2])}")
        return int(ijkFloat[2])

    def oneoneone(self):
        pixel_arr = self.pixelArray()
        shape = pixel_arr.shape()
        zero_array = np.zeros(shape)
        print(zero_array[self.current_slice(), :, :].shape)

    def to_JSON(self):
        url = 'http://127.0.0.1:8000/masks'

        try:
            ijk_points = []
            ras_points = self.get_many_coords()
            ras_serializable = [list(coord) for coord in ras_points]
            for ras in ras_serializable:
                ras = self.ras_to_ijk(ras)
                ijk_points.append(ras)
                #print(f"IJK-точки на сервер: {ijk_points}")
        except Exception as e:
            ijk_points = []

        try:
            ijk_roi = self.ras_to_ijk_roi()
            roi = [list(coord) for coord in roi]
        except Exception as e:
            ijk_roi = []

        try:
            pixel_arr = self.pixelArray()
            pixel_arr = pixel_arr[self.current_slice(), :, :]
            print(f"форма pixel_arr: {pixel_arr.shape}")
            pixel_arr_serializable = pixel_arr.tolist()  # Преобразовываем ndarray в список
        except Exception as e:
            pixel_arr_serializable = []
        try:
            input_label = self.input_label()
            print(f"input_label: {input_label}")
        except Exception as e:
            input_label = []


        data = {
            "points": ijk_points,  # Используем преобразованные данные
            "roi": ijk_roi,
            'pixel_arr': pixel_arr_serializable,
            'input_label': input_label
        }




        print(f'Внутренность JSON: {data["roi"]}, {data["points"]}, {data["input_label"]}')
        print(len(data))




        #print(type(points_serializable), type(roi), type(pixel_arr_serializable))
        username = "root"
        password = "1111"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data, auth=(username, password))

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
        mask_data = self.to_JSON()  # строка
        first_slice = self.current_slice()
        mask = json.loads(mask_data)  # словарь с масками
        mask_points = np.array(mask["mask_points"]).astype(int)
        mask_roi = np.array(mask["mask_roi"]).astype(int)
        #print(f"форма маски roi: {mask_points.shape}")
        shape = self.pixelArray().shape
        print(shape)#размерность dicom  (15,320,320)
        #print(f"shape: {mask_roi.shape}")
        print(f'first_slice {first_slice}')

        if mask_points.any():
            point_array = np.full(shape, 0, dtype=int)
            point_array[first_slice] = mask_points
            mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")  #
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation_prostate")
            segment = segmentationNode.GetSegmentation().AddEmptySegment()
            segment = segmentationNode.GetSegmentation().GetNthSegment(0)
            segment.SetName('prostate')
            segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('prostate')
            slicer.util.updateSegmentBinaryLabelmapFromArray(point_array, segmentationNode, segmentId, mainvolume)

        if mask_roi.any():
            roi_array = np.full(shape, 0, dtype=int)
            roi_array[first_slice] = mask_roi
            mainvolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")  #
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation_prostate")
            segment = segmentationNode.GetSegmentation().AddEmptySegment()
            segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Segment_1')
            slicer.util.updateSegmentBinaryLabelmapFromArray(roi_array, segmentationNode,
                                                             segmentId, mainvolume)








