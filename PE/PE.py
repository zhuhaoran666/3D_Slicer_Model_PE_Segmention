import logging
import os
import glob
import vtk
import numpy as np
import vtk.util.numpy_support as ns
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin





#
# PE
#

class PE(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PE"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Zhu Haoran (SZU)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
"""

        # Additional initialization step after application startup is complete
       
class PEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/PE.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.

        




        #init select
        
        self.ui.MRMLNodeComboBox.setMRMLScene(slicer.mrmlScene)
        self.ui.MRMLNodeComboBox.addEnabled = False
        self.ui.MRMLNodeComboBox.noneEnabled = False
        self.ui.MRMLNodeComboBox.showHidden = False
        self.ui.MRMLNodeComboBox.removeEnabled = True
        self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLScalarVolumeNode']
        node=self.ui.MRMLNodeComboBox.currentNode()
        slicer.util.setSliceViewerLayers(background=node)
        #init model and label path


        ai_model_path=r"C:/Users/Administrator/Desktop/PEsegmention/PE/AImodel/patch-64bce-cut.pth"
        segmention_save_path=r'C:/Users/Administrator/Desktop/PEsegmention/PE/segmention'

        self.ui.PathLineEdit.setCurrentPath(ai_model_path)
        #self.ui.PathLineEdit.filters=ctk.ctkPathLineEdit.Dirs
        self.ui.PathLineEdit_2.setCurrentPath(segmention_save_path)
        self.ui.PathLineEdit_2.filters=ctk.ctkPathLineEdit.Dirs
        self.ui.spinBox.setValue(10)
                
        self.ui.applyButton.clicked.connect(self.applyButton_clicked)
        self.ui.MRMLNodeComboBox.currentNodeChanged.connect(self.onCurrentNodeChanged)


        
        #segname
        #self.ui.lineEdit_3.setText('Coming soon...') 
        self.ui.lineEdit.setText(self.ui.MRMLNodeComboBox.currentNode().GetName())
        

        #slicer.util.loadSegmentation(r'C:\Users\Administrator\Desktop\PEsegmention\PE\segmention\2.seg.nrrd')
        
        #slicer.app.processEvents()
        
        
       
    def applyButton_clicked(self):
        import torch
        import libPredict
        import SimpleITK as sitk
        imagenode=self.ui.MRMLNodeComboBox.currentNode()
        ai_model_path=self.ui.PathLineEdit.currentPath
        save_path=self.ui.PathLineEdit_2.currentPath
        name=self.ui.lineEdit.text
        #print(segmention_name)
        if (os.path.exists(os.path.join(save_path,name+'.nrrd'))) or (os.path.exists(os.path.join(save_path,name+'.seg.nrrd'))):
            print("The file already exists, please change the file name")
        elif(imagenode==None)or(ai_model_path==' '):
            print('please scelet a image and model')
        else:
            node=self.ui.MRMLNodeComboBox.currentNode()
            file_name = name + '.nrrd'
            file_path =os.path.join(save_path,file_name)
            slicer.util.saveNode(node, file_path,properties={"nodeID":node.GetID()})
            #print('gpu:',torch.cuda.is_available())
            GPUID = 0
            patchSize = 64
            numPatchSampleFactor=self.ui.spinBox.value
            #print(numPatchSampleFactor)
            #print("Under calulation,please wait patiently,about 3min.")
            testImageName =os.path.join(save_path,name+'.nrrd')
            outputImageName =os.path.join(save_path,name+'cut.nrrd')
            outputSegName = os.path.join(save_path,name+'.seg.nrrd')
            #print(outputName)
            print(outputImageName,outputSegName)
            libPredict.segmentCurveRes4(ai_model_path, testImageName, outputImageName, outputSegName, GPUid = GPUID, patchSize = patchSize,numPatchSampleFactor=numPatchSampleFactor)
            slicer.util.loadVolume(outputImageName)
            slicer.util.loadSegmentation(outputSegName)
            slicer.app.processEvents()
            Seg=sitk.ReadImage(outputSegName)
            Spacing=Seg.GetSpacing()
            seg=sitk.GetArrayFromImage(Seg)
            Volume=np.count_nonzero(seg)*Spacing[0]*Spacing[1]*Spacing[2]
            self.ui.lineEdit_2.setText(str(Volume/1000)+'cc')

    def onCurrentNodeChanged(self):
        node = self.ui.MRMLNodeComboBox.currentNode()
        slicer.util.setSliceViewerLayers(background=node)
        self.ui.lineEdit.setText(node.GetName())





