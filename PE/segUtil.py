from configparser import InterpolationError
from matplotlib.image import interpolations_names
import numpy
import SimpleITK as sitk
import sys

def adjustGrayImage(img):
    '''normalized by 10-th and 90th percentile to [0,1]

    Note that the training image must be adjusted one by one, not
    hauling all and then adjust. Coz their each's distribution is
    vastly different

    '''

    tmp = img.astype(numpy.float32)
    p10 = numpy.percentile(tmp, 10)
    p90 = numpy.percentile(tmp, 90)

    img = (tmp - p10)/(p90 - p10)

    #img = img.astype(numpy.float32)/255.0


    return img

def adjustLabelImage(mask):
    #mask = mask /255 # my input label is 0/1, not 0-255, so don't have to /255
    mask = mask.astype(numpy.float32)
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0

    return mask

def adjustData(img, mask):

    img = adjustGrayImage(img)
    mask = adjustLabelImage(mask)

    return (img,mask)


################################################################################
# as of 20220520, this function is not used for the hippocampus
# segmetnation. This is because that i can load all the training
# images and their masks to memory, and then just extract patches from
# memory. So i don't have to load a volume every times. This situation
# may change when i have less memory/more training images. So i'm
# keeping this function. But be aware that this may not be updated
def getPatchBatchFromFile(thisImageName, thisMaskName, patchSize, batch_size):
    # in this training patch generator, I do NOT crop the image to be
    # around the object, instead, i randomly pick a location and if
    # the cropped region has anyobject pixel, i'll take it.
    #
    # For other ways of taking samples, including first crop to ROI and then sample, see zzz/data.py

    thisImg = sitk.ReadImage(thisImageName, sitk.sitkFloat64)
    thisMask = sitk.ReadImage(thisMaskName, sitk.sitkFloat64)
    
    thisImg = sitk.GetArrayFromImage(thisImg)
    thisMask = sitk.GetArrayFromImage(thisMask)

    #thisImg, thisMask = adjustData(thisImg, thisMask)


    numChannels = 1 # for gray image

    imageBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    maskBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)

    randIdxForThisBatch = numpy.random.randint(0, n-1, size = 1)
    randomIdx = randIdxForThisBatch[0]

    # #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("This epoch picks the", randomIdx, "-th atlas", flush=True)
    # #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    sz = thisImg.shape

    numValidPatch = 0
    while numValidPatch < batch_size:
        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchSize, size = 1)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchSize, size = 1)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchSize, size = 1)

        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]
        thisTopLeftZ = allPatchTopLeftZ[0]

        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
        if thisLabelImPatch.max() > 0:
            thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]

            imageBatch[numValidPatch, 0, :, :, :] = thisImPatch.astype(numpy.float32)
            maskBatch[numValidPatch, 0, :, :, :] = thisLabelImPatch.astype(numpy.float32)

            numValidPatch += 1

    #print(">>>>>>>", numpy.percentile(imageBatch, 10), numpy.percentile(imageBatch, 90))
    return imageBatch,maskBatch


def mirrorPad(img, patchSize):
    sz = img.GetSize()
    #print(sz, type(sz))

    padUpperBound = [0, 0, 0]

    for i in range(3):
        if sz[i] < patchSize:
            padUpperBound[i] = 10 + patchSize - sz[i]

    mirrorPad = sitk.MirrorPadImageFilter()
    mirrorPad.SetPadUpperBound(padUpperBound)
    newImg = mirrorPad.Execute(img)
    print("old size = ", sz, "new size = ", newImg.GetSize())

    return newImg



def loadAllImagesAndMasks(allImageNames, allMaskImageNames, patchSize):

    n = len(allImageNames)
    print("num of images = ", n, "len(allTrainNamesMask)", len(allMaskImageNames))

    imageAll = []
    maskAll = []
    for it in range(n):
        thisMaskName = allMaskImageNames[it]
        thisImageName = allImageNames[it]

        thisImg = sitk.ReadImage(thisImageName, sitk.sitkFloat64)
        thisMask = sitk.ReadImage(thisMaskName, sitk.sitkFloat64)

        # mirror pad image and mask to the minimal size larger than patchsize
        thisImg = mirrorPad(thisImg, patchSize)
        thisMask = mirrorPad(thisMask, patchSize)
        thisImg = sitk.GetArrayFromImage(thisImg)
        thisMask = sitk.GetArrayFromImage(thisMask)

        print(thisImg.max(), thisMask.max())
        print("Reading", it, "-th image, it's max image pixel:", thisImg.max(), "max label value:", thisMask.max(), end = '')

        thisImg, thisMask = adjustData(thisImg, thisMask)

        #print("after adjustment: ", thisImg.max(), thisMask.max())
        print(" ,       after adjustment max image pixel:", thisImg.max(), "max label:", thisMask.max())
        if(thisMask.max()==0):
            print("drop")
            continue

        imageAll.append(thisImg)
        maskAll.append(thisMask)
    print(len(imageAll))
    return imageAll, maskAll



def getPatchBatchFromMemory(it, imageBatch, maskBatch, patchSize, batch_size, negativePatchRatio = 0):
    # in this training patch generator, I do NOT crop the image to be
    # around the object, instead, i randomly pick a location and if
    # the cropped region has anyobject pixel, i'll take it.
    #
    # For other ways of taking samples, including first crop to ROI and then sample, see zzz/data.py

    thisImg = imageBatch[it]
    thisMask = maskBatch[it]

    #thisImg, thisMask = adjustData(thisImg, thisMask)

    numChannels = 1 # for gray image

    imageBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    maskBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)

    # n = 
    # randIdxForThisBatch = numpy.random.randint(0, n-1, size = 1)
    # randomIdx = randIdxForThisBatch[0]

    # #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("This epoch picks the", randomIdx, "-th atlas", flush=True)
    # #print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    sz = thisImg.shape

    numValidPatch = 0
    while numValidPatch < batch_size:
        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchSize, size = 1)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchSize, size = 1)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchSize, size = 1)

        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]
        thisTopLeftZ = allPatchTopLeftZ[0]

        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]

        # If this patch contains positive pixels (pixel with target),
        # then include this patch in training. Otherwise, if the patch
        # is all-0, then only include that patch with a (small)
        # probability.
        probIncludeNegativePatch = numpy.random.rand(1)
        if thisLabelImPatch.max() > 0 or probIncludeNegativePatch[0] < negativePatchRatio:
            #print("++++++++++++++++++++", thisTopLeftX, thisTopLeftY, thisTopLeftZ)
            thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]

            imageBatch[numValidPatch, 0, :, :, :] = thisImPatch.astype(numpy.float32)
            maskBatch[numValidPatch, 0, :, :, :] = thisLabelImPatch.astype(numpy.float32)

            numValidPatch += 1

    return imageBatch,maskBatch

