import qt
import slicer
import ctk
import vtk
import numpy as np
import json
import requests

class MyNewModule:
    def __init__(self, parent):
        parent.title = "MLMedicine Module"
        parent.categories = ["SAMSegment"]
        parent.dependencies = []
        parent.contributors = ["mnfomenkov@gmail.com, tg: @heeeey_makarena"]
        parent.helpText = "Place point or ROI to create a mask on your DICOM"
        parent.acknowledgementText = "Developed by Makar Fomenkov"
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
        self.current_data = {}  # Для хранения данных между слоями
        self.setup()
        
        if not parent:
            self.parent.show()

    def setup(self):
        collapsibleButton = ctk.ctkCollapsibleButton()
        collapsibleButton.text = "SAM Segmentation"
        
        self.layout.addWidget(collapsibleButton)
        formLayout = qt.QFormLayout(collapsibleButton)
        
        # Scan Button
        self.scanBtn = qt.QPushButton("Scan Scene")
        self.scanBtn.clicked.connect(self.scan_scene)
        formLayout.addRow(self.scanBtn)
        
        # ROI Button
        self.roiBtn = qt.QPushButton("Get ROI")
        self.roiBtn.clicked.connect(self.get_roi_coords)
        formLayout.addRow(self.roiBtn)
        
        # Points Button
        self.pointsBtn = qt.QPushButton("Get Points")
        self.pointsBtn.clicked.connect(self.get_points_coords)
        formLayout.addRow(self.pointsBtn)
        
        # Send Button
        self.sendBtn = qt.QPushButton("Send to Server")
        self.sendBtn.clicked.connect(self.send_to_server)
        formLayout.addRow(self.sendBtn)
        
        # Visualize Button
        self.visualizeBtn = qt.QPushButton("Visualize Mask")
        self.visualizeBtn.clicked.connect(self.visualize_mask)
        formLayout.addRow(self.visualizeBtn)
        
        # Current Slice Label
        self.sliceLabel = qt.QLabel("Current Slice: -")
        formLayout.addRow(self.sliceLabel)
        
        # Positive/Negative Checkbox
        self.positiveCheck = qt.QCheckBox("Positive Point")
        self.positiveCheck.stateChanged.connect(self.set_point_type)
        formLayout.addRow(self.positiveCheck)

    def set_point_type(self, state):
        color = (0, 1, 0) if state else (1, 0, 0)  # Green for positive, red for negative
        fiducialNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        if fiducialNodes:
            fiducialNodes[-1].GetDisplayNode().SetColor(*color)

    def get_current_slice(self):
        layoutManager = slicer.app.layoutManager()
        red = layoutManager.sliceWidget("Red")
        return int(red.sliceLogic().GetSliceOffset())

    def scan_scene(self):
        self.sliceLabel.setText(f"Current Slice: {self.get_current_slice()}")
        
        # Get volume info
        volume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if volume:
            print(f"Volume dimensions: {slicer.util.arrayFromVolume(volume).shape}")
        
        # Get points info
        points = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        print(f"Found {len(points)} point sets")
        
        # Get ROIs info
        rois = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        print(f"Found {len(rois)} ROIs")

    def get_roi_coords(self):
        """Get ROI coordinates for current slice only"""
        slice_idx = self.get_current_slice()
        roi_nodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        
        if not roi_nodes:
            print("No ROIs found")
            return
        
        volume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if not volume:
            print("No volume found")
            return
        
        # Store ROI data for current slice
        self.current_data[slice_idx] = self.current_data.get(slice_idx, {})
        self.current_data[slice_idx]['roi'] = []
        
        for roi in roi_nodes:
            center = np.array(roi.GetCenter())
            size = np.array(roi.GetSize())
            
            # Convert to IJK coordinates
            rasToIjk = vtk.vtkMatrix4x4()
            volume.GetRASToIJKMatrix(rasToIjk)
            
            min_ras = center - size/2
            max_ras = center + size/2
            
            min_ijk = [0]*4
            max_ijk = [0]*4
            rasToIjk.MultiplyPoint(list(min_ras) + [1], min_ijk)
            rasToIjk.MultiplyPoint(list(max_ras) + [1], max_ijk)
            
            # Get 2D ROI for current slice
            roi_coords = [
                int(min_ijk[0]),  # x1
                int(min_ijk[1]),  # y1 
                int(max_ijk[0]),  # x2
                int(max_ijk[1])   # y2
            ]
            
            self.current_data[slice_idx]['roi'].append(roi_coords)
            print(f"ROI for slice {slice_idx}: {roi_coords}")

    def get_points_coords(self):
        """Get points coordinates for current slice only"""
        slice_idx = self.get_current_slice()
        fiducials = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        
        if not fiducials:
            print("No points found")
            return
            
        volume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if not volume:
            print("No volume found")
            return
        
        # Store points data for current slice
        self.current_data[slice_idx] = self.current_data.get(slice_idx, {})
        self.current_data[slice_idx]['points'] = []
        self.current_data[slice_idx]['labels'] = []
        
        for fid in fiducials:
            for i in range(fid.GetNumberOfControlPoints()):
                ras = [0]*3
                fid.GetNthControlPointPositionWorld(i, ras)
                
                # Convert to IJK
                rasToIjk = vtk.vtkMatrix4x4()
                volume.GetRASToIJKMatrix(rasToIjk)
                ijk = [0]*4
                rasToIjk.MultiplyPoint(list(ras) + [1], ijk)
                
                # Check if point is on current slice
                if abs(ijk[2] - slice_idx) < 1:
                    point_2d = [int(ijk[0]), int(ijk[1])]
                    label = 1 if fid.GetDisplayNode().GetColor() == (0,1,0) else 0
                    
                    self.current_data[slice_idx]['points'].append(point_2d)
                    self.current_data[slice_idx]['labels'].append(label)
        
        print(f"Points for slice {slice_idx}: {self.current_data[slice_idx].get('points', [])}")

    def send_to_server(self):
        """Send data for current slice only"""
        slice_idx = self.get_current_slice()
        volume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        
        if not volume:
            print("No volume found")
            return
            
        if slice_idx not in self.current_data:
            print(f"No data for slice {slice_idx}")
            return
        
        # Get current slice image
        pixel_array = slicer.util.arrayFromVolume(volume)[slice_idx]
        
        # Prepare payload
        payload = {
            "slice_index": slice_idx,
            "pixel_arr": pixel_array.tolist(),
            "points": self.current_data[slice_idx].get('points', []),
            "roi": self.current_data[slice_idx].get('roi', []),
            "input_label": self.current_data[slice_idx].get('labels', [])
        }
        
        # Send to server
        try:
            response = requests.post(
                "http://127.0.0.1:8000/masks",
                json=payload,
                auth=("root", "1111"),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            # Store response
            self.current_data[slice_idx]['mask'] = response.json()
            print(f"Mask received for slice {slice_idx}")
            
        except Exception as e:
            print(f"Error sending data: {e}")

    def visualize_mask(self):
        """Visualize mask for current slice"""
        slice_idx = self.get_current_slice()
        
        if slice_idx not in self.current_data or 'mask' not in self.current_data[slice_idx]:
            print(f"No mask available for slice {slice_idx}")
            return
            
        mask_data = self.current_data[slice_idx]['mask']
        volume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        
        if not volume:
            print("No volume found")
            return
            
        # Create segmentation
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        
        # Create segment
        segment_id = segmentation.GetSegmentation().AddEmptySegment(f"Slice_{slice_idx}")
        
        # Prepare mask array
        mask_array = np.zeros(slicer.util.arrayFromVolume(volume).shape, dtype=np.uint8)
        if 'mask_points' in mask_data:
            mask_array[slice_idx] = np.array(mask_data['mask_points'])
        
        # Update segmentation
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            mask_array, segmentation, segment_id, volume
        )
        
        # Set display properties
        display_node = segmentation.GetDisplayNode()
        if display_node:
            display_node.SetVisibility(True)
            display_node.SetOpacity(0.6)
        
        print(f"Mask visualized for slice {slice_idx}")