from pyexpat import model
import sys
import SimpleITK as sitk

import torch
import torch.nn

import numpy

import math

import numpy
import numpy.random
from torch import nn
import torch


#device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
def segment3DPatchBatch(model, inputImagePatchBatchArray, GPUid = 0):

    #print("++++++++++++++++++       ", GPUid)
    # This segments a list of 3D patches. The input
    # inputImagePatchArray is a (#batches, numChannel, patchSideLen,
    # patchSideLen, patchSideLen) numpy array. Next this needs to be reshaped to
    # (#batches, #channels, patchSideLen, patchSideLen, patchSideLen) where and
    # #channels=1 in this case

    #sz = inputImagePatchBatchArray.shape

    #: torch needs chanel first, need to insert a channel axis
    #inputImagePatchBatchArray = numpy.expand_dims(inputImagePatchBatchArray.astype(numpy.float32), axis = 1)

    device = torch.device("cuda:" + str(GPUid) if torch.cuda.is_available() else "cpu")



    #threshold=numpy.where(((inputImagePatchBatchArray>200)&(inputImagePatchBatchArray<500)),1,0)

    #threshold = torch.from_numpy(threshold).float()

    inputImagePatchBatchArray = torch.from_numpy(inputImagePatchBatchArray)

    #inputs=torch.cat([inputImagePatchBatchArray,threshold],dim=1)
    inputs=inputImagePatchBatchArray.to(device)
    inputs = inputs.to(device)

    
    # for i in range(10):
    #: make DNN predictions
    with torch.no_grad():
        results = model(inputs)
        # inputs = torch.cat([inputImagePatchBatchArray,results],dim=1)
    #results=torch.sigmoid(results)
    outputSegBatchArray = results.cpu().numpy()
    #print(results.max())
    #: torch needs chanel first, need to squeez the channel axis
    outputSegBatchArray = outputSegBatchArray[:, 0, :, :, :]

    #outputSegBatchArray = outputSegBatchArray[:, :, :, 0]

    #print("*************  ", outputSegBatchArray.shape, outputSegBatchArray.max())

    return outputSegBatchArray



def segment3DImageRandomSampleDividePrior(model, imageArray, patchSideLen = 64, numPatchSampleFactor = 10,
                                             batch_size = 1, num_segmentation_classes = 1, GPUid = 0):
    sz = imageArray.shape
    print("imageArray.shape = ", sz)
    numChannel = 1 # for gray
    iChannel = 0 # for gray

    # assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen and sz[2] == 3),"Image shape must be >= " + str(patchSideLen) + "-cubed."
    # if sz[2] != numChannel:
    #     print("Only process  image")
    #     exit(-1)

    # the number of random patches is s.t. on average, each pixel is
    # sampled numPatchSampleFactor times. Default is 10
    numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*(sz[2]/patchSideLen)*numPatchSampleFactor)

    #print("numPatchSample = ", numPatchSample)

    # this saves the segmentation result
    segArray = numpy.zeros((num_segmentation_classes, sz[0], sz[1], sz[2]), dtype=numpy.float32)
    priorImage = numpy.zeros((sz[0], sz[1], sz[2]), dtype=numpy.float32)

    patchShape = (patchSideLen, patchSideLen, patchSideLen)
    imagePatchBatch = numpy.zeros((batch_size, numChannel, patchShape[0], patchShape[1], patchShape[2]), dtype=numpy.float32)

    for itPatch in range(0, numPatchSample, batch_size):

        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchShape[2], size = batch_size)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            thisTopLeftZ = allPatchTopLeftZ[itBatch]

            #: ad hoc: 0 in channel axis coz only gray image
            tmp = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]),
                             thisTopLeftY:(thisTopLeftY + patchShape[1]),
                             thisTopLeftZ:(thisTopLeftZ + patchShape[2])]

            imagePatchBatch[itBatch, iChannel, :, :, :] = tmp

        #imagePatchBatch = imagePatchBatch
        segBatch = segment3DPatchBatch(model, imagePatchBatch, GPUid = GPUid)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            thisTopLeftZ = allPatchTopLeftZ[itBatch]

            segArray[:, thisTopLeftX:(thisTopLeftX + patchShape[0]),
                     thisTopLeftY:(thisTopLeftY + patchShape[1]),
                     thisTopLeftZ:(thisTopLeftZ + patchShape[2])] += segBatch[itBatch, :, :, :]
            priorImage[thisTopLeftX:(thisTopLeftX + patchShape[0]),
                       thisTopLeftY:(thisTopLeftY + patchShape[1]),
                       thisTopLeftZ:(thisTopLeftZ + patchShape[2])] += numpy.ones((patchShape[0], patchShape[1], patchShape[2]))

    for it in range(num_segmentation_classes):
        segArray[it, :, :, :] /= (priorImage + numpy.finfo(numpy.float32).eps)
        segArray[it, :, :, :]*=100
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # segArray contains multiple channels of output. The 1-st is the
    # corresponding output for the 1st object.
    print(numpy.max(segArray[0,:, :, :]),numpy.min(segArray[0,:, :, :]))
    """ max=numpy.max(segArray[0,:, :, :])
    min=numpy.min(segArray[0,:, :, :]) """
    segArray=numpy.where(segArray>50,1,0)
    # print(numpy.max(segArray[0,:, :, :]))
    # print(numpy.min(segArray[0,:, :, :]))
    outputSegArrayOfObject1 = segArray[0, :, :, :]
    return outputSegArrayOfObject1




import segUtil
import math
import numpy as np

def segmentCurveRes4(modelName, inputImageName, outputImageName , outputSegName, GPUid = 0, patchSize = 64, numPatchSampleFactor=10):
    net = torch.load(modelName, map_location=torch.device('cuda:' + str(GPUid)))
    net.eval()

    testIm = worker_predict(inputImageName,outputImageName)

   
    
    testIm = segUtil.mirrorPad(testIm, patchSize) #: so the image is larger than patchSize in all axises
    testImg = sitk.GetArrayFromImage(testIm)
    print(testImg.max())
    testImg = segUtil.adjustGrayImage(testImg)
    print(testImg.max())


    testImgSeg = segment3DImageRandomSampleDividePrior(model = net, imageArray = testImg, patchSideLen = patchSize, numPatchSampleFactor = numPatchSampleFactor, batch_size = 1, num_segmentation_classes = 1, GPUid = GPUid)

    outputSeg = sitk.GetImageFromArray(testImgSeg.astype(numpy.uint8), isVector=False)
    outputSeg.SetSpacing(testIm.GetSpacing())
    outputSeg.SetOrigin(testIm.GetOrigin())
    outputSeg.SetDirection(testIm.GetDirection())
    sitk.WriteImage(outputSeg, outputSegName, True)



def worker_predict(Imagename,outputImageName):
    
    
    ctImage = sitk.ReadImage(Imagename)
    print('old:')
    print('Imagesize:',ctImage.GetSize())
    print('ImageSpacing',ctImage.GetSpacing())
    airMask = segmentAirFromChestCT(ctImage)
    
    airMask = sitk.BinaryFillhole(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryDilate(airMask)
    airMask = sitk.BinaryErode(airMask)
    airMask = sitk.BinaryErode(airMask)
    airMask = sitk.BinaryErode(airMask)
    airMask = sitk.BinaryErode(airMask)
    airMask = sitk.BinaryErode(airMask)
    airMask = sitk.BinaryErode(airMask)
    Image=sitk.GetArrayFromImage(ctImage)
    
    BoneSeg=sitk.GetArrayFromImage(airMask)
    start1,end1=0,0
    start2,end2=0,0
    start0,end0=0,0
    for i in range(BoneSeg.shape[0]):
        count=np.count_nonzero(BoneSeg[i,:,:])
        if count!=0:
            start0=i
            break
    for i in range(BoneSeg.shape[0]-1,-1,-1):
        count=np.count_nonzero(BoneSeg[i,:,:])
        if count!=0:
            end0=i
            break
    for i in range(BoneSeg.shape[1]):
        count=np.count_nonzero(BoneSeg[:,i,:])
        if count!=0:
            start1=i
            break
    for i in range(BoneSeg.shape[1]-1,-1,-1):
        count=np.count_nonzero(BoneSeg[:,i,:])
        if count!=0:
            end1=i
            break
    for i in range(BoneSeg.shape[2]):
        count=np.count_nonzero(BoneSeg[:,:,i])
        if count!=0:
            start2=i
            break
    for i in range(BoneSeg.shape[2]-1,-1,-1):
        count=np.count_nonzero(BoneSeg[:,:,i])
        if count!=0:
            end2=i
            break
    
    Image=Image[start0:end0+1,start1:end1+1,start2:end2+1]

    outputImage = sitk.GetImageFromArray(Image, isVector=False)
    outputImage.SetSpacing(ctImage.GetSpacing())
    outputImage.SetOrigin(ctImage.GetOrigin())
    outputImage.SetDirection(ctImage.GetDirection())

    print(Image.shape)
    print(outputImage.GetSpacing())
    outputImage=resample_image(outputImage,label=False)
    print(outputImage.GetSpacing())
    print(sitk.GetArrayFromImage(outputImage).shape)

    print('new:')
    print('Imagesize:',ctImage.GetSize())
    print('ImageSpacing',ctImage.GetSpacing())

    sitk.WriteImage(outputImage,outputImageName,True)
    return outputImage
def resample_image(itk_image,spacing=[1,1,1],label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_spacing=spacing
    #for i in original_spacing:
    #    out_spacing.append(i*spacing_multiple)
    
    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if label==False:
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    
    return resample.Execute(itk_image)
def segmentAirFromChestCT(ctImage):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find the air region
    thld = 250
    b = ctImage > thld
    c = sitk.ConnectedComponent(b)
    f = sitk.RelabelComponentImageFilter()
    #f.SetMinimumObjectSize(1000000)
    cc = f.Execute(c)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Among all the regions, find the one whose center is closest to
    # the image center. Apparently, this condition dictates that the
    # image must be chest CT: center of mass is somewhere in the cente
    # of the lung.
    #
    #print("------------")
    ccArray = sitk.GetArrayFromImage(cc)
    #print("    ", ccArray.shape)
    c0 = ccArray.shape[0]/2.0
    c1 = ccArray.shape[1]/2.0
    c2 = ccArray.shape[2]/2.0

    candidateLabels = [1,2, 3, 4]
    r = [None]*len(candidateLabels)

    for it in range(len(candidateLabels)):
        itLabel = candidateLabels[it]

        idx = numpy.where(ccArray == itLabel)

        m0 = numpy.mean(idx[0])
        m1 = numpy.mean(idx[1])
        m2 = numpy.mean(idx[2])
        # s0 = numpy.std(idx[0])
        # s1 = numpy.std(idx[1])
        # s2 = numpy.std(idx[2])

        r[it] = math.sqrt(((m0 - c0)**2 + (m1 - c1)**2 + (m2 - c2)**2))
        #print("    ", itLabel, (m0, m1, m2), (s0, s1, s2), r[it])


    lungLabel = candidateLabels[numpy.argmin(r)]
    #print("       lung's label =", lungLabel)

    airMask = (cc == lungLabel)

    return airMask